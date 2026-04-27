// track_split_axelera.cpp — SiamRPN++ tracker with Axelera Metis backbone (C++)
// template_encoder + search_encoder on Axelera Metis (axruntime)
// xcorr split into: kernel-proj (ORT, init only) + search-proj (ORT) +
//                   dw-xcorr (C++ AVX2) + head (ORT)
// particle filter for robust tracking
//
// Optimizations vs original:
//   1. -mavx2 -mfma in Makefile (vectorizes all float loops)
//   2. __restrict__ on quantize/dequantize pointer args (removes aliasing barriers)
//   3. xcorr_head.onnx split: 6 dynamic grouped convs replaced by C++ AVX2 dw-xcorr
//      kernel projections only re-run at template updates (every tmpl_freq frames)
//
// Usage:
//   ./track_split_axelera_cpp \
//     --template_encoder /path/to/compiled_template_v2/compiled_model/model.json \
//     --search_encoder   /path/to/compiled_search_v2/compiled_model/model.json \
//     --kernel_proj      /path/to/xcorr_kernel_proj.onnx \
//     --search_proj      /path/to/xcorr_search_proj.onnx \
//     --head             /path/to/xcorr_head_only.onnx \
//     --video            /path/to/video.mp4 \
//     --output           /path/to/output.mp4 \
//     [--init_bbox       348,147,38,84]   (omit to select interactively with mouse)
//     [--ratios          0.33,0.5,1,2,3]  (default: 0.37,0.56,0.79,1.11,2.26 for IR model)
//     [--score_csv       /path/to/scores.csv]  (per-frame scores for comparison)
//     [--max_frames      N]
//     [--display]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <axruntime/axruntime.h>

using namespace std::chrono;

// ─── Numpy-compatible MT19937 + Box-Muller RNG ────────────────────────────────
struct NumpyRNG {
    static const int MT_N = 624;
    static const int MT_M = 397;
    uint32_t mt[624];
    int      mti;
    double   gauss_val;
    bool     has_gauss;

    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < MT_N; i++)
            mt[i] = 1812433253u * (mt[i-1] ^ (mt[i-1] >> 30)) + (uint32_t)i;
        mti = MT_N + 1;
    }

    void seed(uint32_t s) {
        init_genrand(s);
        has_gauss = false; gauss_val = 0.0;
    }

    uint32_t rk_ulong() {
        static const uint32_t MATRIX_A = 0x9908b0dfu;
        static const uint32_t UPPER    = 0x80000000u;
        static const uint32_t LOWER    = 0x7fffffffu;
        if (mti >= MT_N) {
            int kk = 0;
            for (; kk < MT_N - MT_M; kk++) {
                uint32_t y = (mt[kk] & UPPER) | (mt[kk+1] & LOWER);
                mt[kk] = mt[kk+MT_M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (; kk < MT_N - 1; kk++) {
                uint32_t y = (mt[kk] & UPPER) | (mt[kk+1] & LOWER);
                mt[kk] = mt[kk-(MT_N-MT_M)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            uint32_t y = (mt[MT_N-1] & UPPER) | (mt[0] & LOWER);
            mt[MT_N-1] = mt[MT_M-1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            mti = 0;
        }
        uint32_t y = mt[mti++];
        y ^= y >> 11;
        y ^= (y <<  7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    double rk_double() {
        long a = (long)(rk_ulong() >> 5);
        long b = (long)(rk_ulong() >> 6);
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    double rk_gauss() {
        if (has_gauss) {
            has_gauss = false;
            double v = gauss_val; gauss_val = 0.0;
            return v;
        }
        double x1, x2, r2, f;
        do {
            x1 = 2.0 * rk_double() - 1.0;
            x2 = 2.0 * rk_double() - 1.0;
            r2 = x1*x1 + x2*x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_val = f * x1; has_gauss = true;
        return f * x2;
    }
};

// ─── Config ──────────────────────────────────────────────────────────────────
struct Config {
    int   stride          = 8;
    std::vector<float> ratios = {0.37f, 0.56f, 0.79f, 1.11f, 2.26f};
    float scale           = 8.0f;
    int   anchor_num      = 5;
    int   score_size      = 25;

    int   exemplar_size   = 127;
    int   instance_size   = 255;
    float context_amount  = 0.5f;
    float penalty_k       = 0.05f;
    float window_influence = 0.42f;
    float lr              = 0.38f;

    int   n_particles     = 500;
    float sigma_pos       = 4.0f;
    float sigma_vel       = 1.5f;
    float sigma_scale     = 0.02f;
    float max_vel         = 25.0f;
    float tau             = 0.15f;
    float roughen_std     = 0.5f;

    int   tmpl_freq       = 30;
    float tmpl_alpha      = 0.25f;
    float tmpl_min_psr    = 6.0f;

    std::string tmpl_enc_path;    // compiled_model/model.json
    std::string srch_enc_path;    // compiled_model/model.json
    std::string kernel_proj_path; // xcorr_kernel_proj.onnx (CPU, run at init only)
    std::string search_proj_path; // xcorr_search_proj.onnx (CPU, every frame)
    std::string head_path;        // xcorr_head_only.onnx   (CPU, every frame)
    std::string video_path;
    std::string output_path;
    std::string score_csv_path;   // optional per-frame CSV output
    int   bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;  // 0 = select interactively
    int   aipu_cores = 4;
    int   max_frames = -1;
    bool  display = false;
};

// ─── Tensor helper ───────────────────────────────────────────────────────────
struct Tensor {
    std::vector<float>   data;
    std::vector<int64_t> shape;
    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

// ─── Anchor generation ───────────────────────────────────────────────────────
std::vector<float> generate_anchors(const Config& cfg) {
    int n_pos = cfg.score_size * cfg.score_size;
    int total = cfg.anchor_num * n_pos;
    std::vector<float> out(total * 4);

    float a0x = (float)(cfg.instance_size / 2) - (float)(cfg.score_size / 2) * cfg.stride;

    int k = 0;
    for (float ratio : cfg.ratios) {
        int ws = (int)std::sqrt((float)(cfg.stride * cfg.stride) / ratio);
        int hs = (int)(ws * ratio);
        float aw = ws * cfg.scale;
        float ah = hs * cfg.scale;
        for (int i = 0; i < cfg.score_size; i++) {
            for (int j = 0; j < cfg.score_size; j++) {
                int idx = k * n_pos + i * cfg.score_size + j;
                out[idx*4 + 0] = a0x + j * cfg.stride;
                out[idx*4 + 1] = a0x + i * cfg.stride;
                out[idx*4 + 2] = aw;
                out[idx*4 + 3] = ah;
            }
        }
        k++;
    }
    return out;
}

// ─── Cosine window ───────────────────────────────────────────────────────────
std::vector<float> make_cosine_window(const Config& cfg) {
    int sz = cfg.score_size;
    std::vector<float> h(sz);
    for (int i = 0; i < sz; i++)
        h[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (sz - 1));

    std::vector<float> win(cfg.anchor_num * sz * sz);
    for (int k = 0; k < cfg.anchor_num; k++)
        for (int i = 0; i < sz; i++)
            for (int j = 0; j < sz; j++)
                win[k*sz*sz + i*sz + j] = h[i] * h[j];
    return win;
}

// ─── Image preprocessing ─────────────────────────────────────────────────────
std::vector<float> get_subwindow_tensor(const cv::Mat& img,
                                         double cx, double cy,
                                         int model_sz, float original_sz,
                                         const cv::Scalar& avg_chans) {
    double c = (original_sz + 1.0) / 2.0;
    int xmin = (int)std::floor(cx - c + 0.5);
    int xmax = xmin + (int)original_sz - 1;
    int ymin = (int)std::floor(cy - c + 0.5);
    int ymax = ymin + (int)original_sz - 1;

    int lpad = std::max(0, -xmin);
    int tpad = std::max(0, -ymin);
    int rpad = std::max(0, xmax - img.cols + 1);
    int bpad = std::max(0, ymax - img.rows + 1);

    xmin += lpad; xmax += lpad;
    ymin += tpad; ymax += tpad;

    cv::Mat padded;
    if (lpad || tpad || rpad || bpad)
        cv::copyMakeBorder(img, padded, tpad, bpad, lpad, rpad,
                           cv::BORDER_CONSTANT, avg_chans);
    else
        padded = img;

    cv::Mat crop = padded(cv::Rect(xmin, ymin,
                                    xmax - xmin + 1,
                                    ymax - ymin + 1)).clone();
    cv::Mat rsz;
    if (model_sz != (int)original_sz)
        cv::resize(crop, rsz, cv::Size(model_sz, model_sz));
    else
        rsz = crop;

    std::vector<float> t(3 * model_sz * model_sz);
    for (int c2 = 0; c2 < 3; c2++)
        for (int i = 0; i < model_sz; i++)
            for (int j = 0; j < model_sz; j++)
                t[c2*model_sz*model_sz + i*model_sz + j] =
                    (float)rsz.at<cv::Vec3b>(i, j)[c2];
    return t;
}

// ─── Score/bbox decoding ─────────────────────────────────────────────────────
std::vector<float> decode_scores(const float* cls, int n_anch, int n_pos) {
    int n = n_anch * n_pos;
    std::vector<float> s(n);
    for (int k = 0; k < n_anch; k++) {
        for (int p = 0; p < n_pos; p++) {
            int idx = k * n_pos + p;
            float s0 = cls[idx], s1 = cls[n + idx];
            float m = std::max(s0, s1);
            float e0 = std::exp(s0 - m), e1 = std::exp(s1 - m);
            s[idx] = e1 / (e0 + e1);
        }
    }
    return s;
}

std::vector<std::array<float,4>> decode_bbox(const float* loc, int n,
                                              const std::vector<float>& anch) {
    std::vector<std::array<float,4>> b(n);
    for (int i = 0; i < n; i++) {
        float dx = loc[0*n+i], dy = loc[1*n+i];
        float dw = loc[2*n+i], dh = loc[3*n+i];
        float acx = anch[i*4], acy = anch[i*4+1];
        float aw  = anch[i*4+2], ah = anch[i*4+3];
        b[i] = { dx*aw + acx, dy*ah + acy,
                 std::exp(dw)*aw, std::exp(dh)*ah };
    }
    return b;
}

static float sz_wh(float w, float h) {
    float pad = (w + h) * 0.5f;
    return std::sqrt((w + pad) * (h + pad));
}

struct BestResult {
    float cx, cy, w, h, score, lr, psr;
    std::vector<float> score_map_25x25;
};

BestResult select_best(const std::vector<float>& scores,
                        const std::vector<std::array<float,4>>& bboxes,
                        const std::vector<float>& window,
                        float scale_z, float prev_w, float prev_h,
                        const Config& cfg) {
    int n = (int)scores.size();
    float psz = sz_wh(prev_w * scale_z, prev_h * scale_z);
    float pr  = prev_w / (prev_h + 1e-6f);
    std::vector<float> pscore(n), penalty(n);

    for (int i = 0; i < n; i++) {
        float sc = std::max(sz_wh(bboxes[i][2], bboxes[i][3]) / (psz + 1e-6f),
                            psz / (sz_wh(bboxes[i][2], bboxes[i][3]) + 1e-6f));
        float rc = std::max(bboxes[i][2] / (bboxes[i][3] + 1e-6f) / (pr + 1e-6f),
                            pr / (bboxes[i][2] / (bboxes[i][3] + 1e-6f) + 1e-6f));
        penalty[i] = std::exp(-(sc * rc - 1.0f) * cfg.penalty_k);
        pscore[i]  = penalty[i] * scores[i] * (1 - cfg.window_influence)
                   + window[i] * cfg.window_influence;
    }

    int best = (int)(std::max_element(pscore.begin(), pscore.end()) - pscore.begin());
    float best_score = scores[best];
    float best_lr    = penalty[best] * best_score * cfg.lr;

    int sz = cfg.score_size;
    std::vector<float> sm(sz * sz, 0);
    for (int k = 0; k < cfg.anchor_num; k++)
        for (int p = 0; p < sz*sz; p++)
            sm[p] = std::max(sm[p], scores[k*sz*sz + p]);

    float fg_max = *std::max_element(scores.begin(), scores.end());
    float fg_sum = 0; for (float v : scores) fg_sum += v;
    float fg_mean = fg_sum / (float)scores.size();
    float fg_sq   = 0; for (float v : scores) fg_sq += (v - fg_mean)*(v - fg_mean);
    float fg_std  = std::sqrt(fg_sq / (float)scores.size() + 1e-12f);
    float psr     = (fg_max - fg_mean) / (fg_std + 1e-6f);

    return { bboxes[best][0] / scale_z, bboxes[best][1] / scale_z,
             bboxes[best][2] / scale_z, bboxes[best][3] / scale_z,
             best_score, best_lr, psr, sm };
}

// ─── Particle filter ─────────────────────────────────────────────────────────
struct Particle { float cx, cy, vx, vy, w, h; };

std::vector<int> systematic_resample(const std::vector<float>& w, NumpyRNG& nrng) {
    int N = (int)w.size();
    float sum = 0; for (float x : w) sum += x;
    float inv = 1.0f / (sum + 1e-30f);

    std::vector<float> cs(N);
    cs[0] = w[0] * inv;
    for (int i = 1; i < N; i++) cs[i] = cs[i-1] + w[i] * inv;
    cs[N-1] = 1.0f;

    float start = (float)nrng.rk_double() / N;
    std::vector<int> idx(N);
    int i = 0, j = 0;
    while (i < N) {
        float pos = start + (float)i / N;
        if (pos < cs[j]) idx[i++] = j;
        else j = std::min(j+1, N-1);
    }
    return idx;
}

// ─── Axelera Metis encoder ────────────────────────────────────────────────────
struct AxeleraEncoder {
    axrContext*       ctx     = nullptr;
    axrModel*         model   = nullptr;
    axrConnection*    conn    = nullptr;
    axrModelInstance* inst    = nullptr;

    std::vector<axrTensorInfo> in_infos;
    std::vector<axrTensorInfo> out_infos;
    std::vector<std::vector<int8_t>> in_bufs;
    std::vector<std::vector<int8_t>> out_bufs;

    void init(axrContext* c, const char* model_path, int aipu_cores) {
        ctx   = c;
        model = axr_load_model(ctx, model_path);
        if (!model) {
            fprintf(stderr, "axr_load_model failed for: %s\n  error: %s\n",
                    model_path, axr_last_error_string(AXR_OBJECT(ctx)));
            exit(1);
        }

        conn = axr_device_connect(ctx, nullptr, 1, nullptr);
        if (!conn) {
            fprintf(stderr, "axr_device_connect failed: %s\n",
                    axr_last_error_string(AXR_OBJECT(ctx)));
            exit(1);
        }

        char prop_str[128];
        snprintf(prop_str, sizeof(prop_str), "aipu_cores=%d\nnum_sub_devices=1", aipu_cores);
        axrProperties* mi_props = axr_create_properties(ctx, prop_str);
        inst = axr_load_model_instance(conn, model, mi_props);
        axr_destroy(AXR_OBJECT(mi_props));
        if (!inst) {
            fprintf(stderr, "axr_load_model_instance failed: %s\n",
                    axr_last_error_string(AXR_OBJECT(ctx)));
            exit(1);
        }

        // Cache input tensor infos and allocate buffers
        size_t ni = axr_num_model_inputs(model);
        for (size_t i = 0; i < ni; i++) {
            axrTensorInfo info = axr_get_model_input(model, i);
            in_infos.push_back(info);
            in_bufs.emplace_back(axr_tensor_size(&info), (int8_t)info.zero_point);
        }

        // Cache output tensor infos and allocate buffers
        size_t no = axr_num_model_outputs(model);
        for (size_t i = 0; i < no; i++) {
            axrTensorInfo info = axr_get_model_output(model, i);
            out_infos.push_back(info);
            out_bufs.emplace_back(axr_tensor_size(&info), 0);
        }
    }

    // Quantize float32 NCHW → int8 NHWC (padded), filling output buffer in-place.
    // __restrict__ on nchw removes GCC aliasing barriers so the inner c-loop
    // (unit-stride writes to buf) can be auto-vectorized with -mavx2.
    void quantize_into(const float* __restrict__ nchw, int C_in, int H_in, int W_in,
                       std::vector<int8_t>& buf, const axrTensorInfo& info) {
        int pH = (int)info.dims[1];
        int pW = (int)info.dims[2];
        int pC = (int)info.dims[3];

        int h0 = (int)info.padding[1][0];
        int w0 = (int)info.padding[2][0];
        int c0 = (int)info.padding[3][0];

        float scale = (float)info.scale;
        int   zp    = info.zero_point;

        std::fill(buf.begin(), buf.end(), (int8_t)zp);

        int8_t* __restrict__ dst = buf.data();
        for (int h = 0; h < H_in; h++) {
            for (int w = 0; w < W_in; w++) {
                int8_t* __restrict__ row = dst + (h+h0)*pW*pC + (w+w0)*pC + c0;
                #pragma GCC ivdep
                for (int c = 0; c < C_in; c++) {
                    float val = nchw[c * H_in * W_in + h * W_in + w];
                    float q   = std::round(val / scale + zp);
                    q = std::max(-128.0f, std::min(127.0f, q));
                    row[c] = (int8_t)(int)q;
                }
            }
        }
    }

    // Dequantize int8 NHWC (padded) → float32 NCHW (unpadded), returning a Tensor.
    // __restrict__ removes aliasing barriers; inner w-loop (unit-stride reads from
    // buf + stride-pC writes to out) benefits from -mavx2 scatter/gather.
    Tensor dequantize(const std::vector<int8_t>& buf, const axrTensorInfo& info) const {
        int pH = (int)info.dims[1];
        int pW = (int)info.dims[2];
        int pC = (int)info.dims[3];

        int h0 = (int)info.padding[1][0], h1 = (int)info.padding[1][1];
        int w0 = (int)info.padding[2][0], w1 = (int)info.padding[2][1];
        int c0 = (int)info.padding[3][0], c1 = (int)info.padding[3][1];

        int H = pH - h0 - h1;
        int W = pW - w0 - w1;
        int C = pC - c0 - c1;

        float scale = (float)info.scale;
        int   zp    = info.zero_point;

        Tensor out;
        out.shape = {1, (int64_t)C, (int64_t)H, (int64_t)W};
        out.data.resize(C * H * W);

        const int8_t* __restrict__ src = buf.data();
        float*        __restrict__ dst = out.data.data();
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                const int8_t* __restrict__ row = src + (h+h0)*pW*pC + (w+w0)*pC + c0;
                #pragma GCC ivdep
                for (int c = 0; c < C; c++)
                    dst[c*H*W + h*W + w] = ((float)(int)row[c] - zp) * scale;
            }
        }
        return out;
    }

    // Per-call timing accumulators (ms) — inspectable from outside.
    double t_quant = 0, t_run = 0, t_dequant = 0;
    long   n_runs  = 0;

    // Run: float32 NCHW input → list of float32 NCHW outputs. Returns (outputs, total_ms).
    std::pair<std::vector<Tensor>, double> run(const std::vector<float>& nchw,
                                                int C_in, int H_in, int W_in) {
        auto ms_since = [](auto t) {
            return duration_cast<microseconds>(
                high_resolution_clock::now() - t).count() / 1000.0;
        };

        auto t0 = high_resolution_clock::now();
        quantize_into(nchw.data(), C_in, H_in, W_in, in_bufs[0], in_infos[0]);
        t_quant += ms_since(t0);

        std::vector<axrArgument> in_args(in_bufs.size());
        for (size_t i = 0; i < in_bufs.size(); i++)
            in_args[i] = { in_bufs[i].data(), -1, 0, in_bufs[i].size() };

        std::vector<axrArgument> out_args(out_bufs.size());
        for (size_t i = 0; i < out_bufs.size(); i++)
            out_args[i] = { out_bufs[i].data(), -1, 0, out_bufs[i].size() };

        auto t1 = high_resolution_clock::now();
        axrResult r = axr_run_model_instance(inst,
                                              in_args.data(),  in_args.size(),
                                              out_args.data(), out_args.size());
        if (r != AXR_SUCCESS) {
            fprintf(stderr, "axr_run_model_instance failed: %s\n", axr_error_string(r));
            exit(1);
        }
        t_run += ms_since(t1);

        auto t2 = high_resolution_clock::now();
        std::vector<Tensor> outputs;
        outputs.reserve(out_bufs.size());
        for (size_t i = 0; i < out_bufs.size(); i++)
            outputs.push_back(dequantize(out_bufs[i], out_infos[i]));
        t_dequant += ms_since(t2);

        ++n_runs;
        return { std::move(outputs), ms_since(t0) };
    }

    void print_latency_breakdown(const char* label) const {
        if (!n_runs) return;
        double total = t_quant + t_run + t_dequant;
        fprintf(stderr, "\n── %s latency breakdown (%ld runs) ──\n", label, n_runs);
        fprintf(stderr, "  quantize (CPU):         %5.2f ms/call  (%4.1f%%)\n",
                t_quant/n_runs,   t_quant/total*100);
        fprintf(stderr, "  axr_run_model_instance: %5.2f ms/call  (%4.1f%%)  ← AIPU kernel + DMA\n",
                t_run/n_runs,     t_run/total*100);
        fprintf(stderr, "  dequantize (CPU):       %5.2f ms/call  (%4.1f%%)\n",
                t_dequant/n_runs, t_dequant/total*100);
        fprintf(stderr, "  total:                  %5.2f ms/call\n", total/n_runs);
    }
};

// Center-crop spatial dims of a (1, C, Sp, Sp) tensor to (1, C, target, target).
static Tensor center_crop(const Tensor& t, int target) {
    int C  = (int)t.shape[1];
    int Sp = (int)t.shape[2];
    if (Sp == target) return t;
    int c0 = (Sp - target) / 2;
    Tensor out;
    out.shape = {1, (int64_t)C, (int64_t)target, (int64_t)target};
    out.data.resize(C * target * target);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < target; h++)
            for (int w = 0; w < target; w++)
                out.data[c*target*target + h*target + w] =
                    t.data[c*Sp*Sp + (h+c0)*Sp + (w+c0)];
    return out;
}

// Run template encoder and center-crop outputs to 7×7 if needed.
static std::pair<std::vector<Tensor>, double>
run_template_encoder(AxeleraEncoder& enc, const std::vector<float>& tmpl_t, int H, int W) {
    auto [raw_feats, ms] = enc.run(tmpl_t, 3, H, W);
    std::vector<Tensor> zf;
    zf.reserve(raw_feats.size());
    for (auto& f : raw_feats) {
        int sp = (int)f.shape[2];
        zf.push_back(sp > 7 ? center_crop(f, 7) : f);
    }
    return { std::move(zf), ms };
}

// ─── Depthwise XCorr (AVX2-vectorized via __restrict__ + ivdep) ─────────────
// search: (1,C,H,W), kernel: (1,C,kH,kW) → (1,C,H-kH+1,W-kW+1)
// Same kernel used for both the LT tracker and this PF tracker.
// The kernel_proj ONNX outputs shape (C,1,kH,kW) — identical memory layout
// to (1,C,kH,kW), so we just set .shape[1]=C when loading.
static Tensor dw_xcorr(const Tensor& search, const Tensor& kernel) {
    const int C  = (int)search.shape[1];
    const int H  = (int)search.shape[2], W  = (int)search.shape[3];
    const int kH = (int)kernel.shape[2], kW = (int)kernel.shape[3];
    const int oH = H - kH + 1, oW = W - kW + 1;

    Tensor out;
    out.shape = {1, (int64_t)C, (int64_t)oH, (int64_t)oW};
    out.data.assign((size_t)C * oH * oW, 0.0f);

    const float* __restrict__ X = search.data.data();
    const float* __restrict__ K = kernel.data.data();
    float*       __restrict__ O = out.data.data();

    for (int c = 0; c < C; ++c) {
        const float* __restrict__ Xc = X + c * H * W;
        const float* __restrict__ Kc = K + c * kH * kW;
        float*       __restrict__ Oc = O + c * oH * oW;
        for (int dr = 0; dr < kH; ++dr) {
            for (int dc = 0; dc < kW; ++dc) {
                const float kval = Kc[dr * kW + dc];
                for (int r = 0; r < oH; ++r) {
                    const float* __restrict__ s = Xc + (r + dr) * W + dc;
                    float*       __restrict__ d = Oc + r * oW;
                    #pragma GCC ivdep
                    for (int oc = 0; oc < oW; ++oc)
                        d[oc] += kval * s[oc];
                }
            }
        }
    }
    return out;
}

// ─── ORT helpers (xcorr_head CPU) ───────────────────────────────────────────
static Ort::Value make_ort_value(Ort::MemoryInfo& mem, std::vector<float>& d,
                                  std::vector<int64_t> shape) {
    return Ort::Value::CreateTensor<float>(mem, d.data(), d.size(),
                                           shape.data(), shape.size());
}

static std::vector<Ort::Value> run_session(
    Ort::Session& sess,
    const std::vector<const char*>& in_names,
    std::vector<Ort::Value>& in_vals,
    const std::vector<const char*>& out_names)
{
    return sess.Run(Ort::RunOptions{nullptr},
                    in_names.data(), in_vals.data(), in_names.size(),
                    out_names.data(), out_names.size());
}

static std::vector<std::string> get_io_names(Ort::Session& s, bool input) {
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n = input ? s.GetInputCount() : s.GetOutputCount();
    std::vector<std::string> names(n);
    for (size_t i = 0; i < n; i++) {
        auto p = input ? s.GetInputNameAllocated(i, alloc)
                       : s.GetOutputNameAllocated(i, alloc);
        names[i] = p.get();
    }
    return names;
}

static Tensor ort_to_tensor(Ort::Value& v) {
    auto info = v.GetTensorTypeAndShapeInfo();
    Tensor t;
    t.shape = info.GetShape();
    const float* ptr = v.GetTensorData<float>();
    t.data.assign(ptr, ptr + info.GetElementCount());
    return t;
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--template_encoder" && i+1 < argc) cfg.tmpl_enc_path    = argv[++i];
        else if (a == "--search_encoder"   && i+1 < argc) cfg.srch_enc_path    = argv[++i];
        else if (a == "--kernel_proj"      && i+1 < argc) cfg.kernel_proj_path = argv[++i];
        else if (a == "--search_proj"      && i+1 < argc) cfg.search_proj_path = argv[++i];
        else if (a == "--head"             && i+1 < argc) cfg.head_path        = argv[++i];
        else if (a == "--video"            && i+1 < argc) cfg.video_path       = argv[++i];
        else if (a == "--output"           && i+1 < argc) cfg.output_path      = argv[++i];
        else if (a == "--score_csv"        && i+1 < argc) cfg.score_csv_path   = argv[++i];
        else if (a == "--aipu_cores"       && i+1 < argc) cfg.aipu_cores       = std::stoi(argv[++i]);
        else if (a == "--max_frames"       && i+1 < argc) cfg.max_frames       = std::stoi(argv[++i]);
        else if (a == "--display")                        cfg.display = true;
        else if (a == "--init_bbox"        && i+1 < argc)
            sscanf(argv[++i], "%d,%d,%d,%d",
                   &cfg.bbox_x, &cfg.bbox_y, &cfg.bbox_w, &cfg.bbox_h);
        else if (a == "--ratios"           && i+1 < argc) {
            cfg.ratios.clear();
            std::string tok, raw = argv[++i];
            for (char& c : raw) if (c == ',') c = ' ';
            std::istringstream ss(raw);
            while (ss >> tok) cfg.ratios.push_back(std::stof(tok));
            cfg.anchor_num = (int)cfg.ratios.size();
        }
    }

    auto anchors    = generate_anchors(cfg);
    auto cosine_win = make_cosine_window(cfg);
    int total_anch  = cfg.anchor_num * cfg.score_size * cfg.score_size;

    // ── Axelera context + encoders ───────────────────────────────────────────
    axrContext* axr_ctx = axr_create_context();
    if (!axr_ctx) { fprintf(stderr, "axr_create_context failed\n"); return 1; }

    AxeleraEncoder tmpl_enc, srch_enc;
    std::cout << "Loading template_encoder from Metis..." << std::flush;
    tmpl_enc.init(axr_ctx, cfg.tmpl_enc_path.c_str(), cfg.aipu_cores);
    std::cout << " done (" << axr_num_model_outputs(tmpl_enc.model)
              << " outputs)\n";

    std::cout << "Loading search_encoder from Metis..." << std::flush;
    srch_enc.init(axr_ctx, cfg.srch_enc_path.c_str(), cfg.aipu_cores);
    std::cout << " done (" << axr_num_model_outputs(srch_enc.model)
              << " outputs)\n";

    // ── ORT sessions (CPU) ───────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "siamrpn_axelera");
    Ort::SessionOptions cpu_opts;
    cpu_opts.SetIntraOpNumThreads(4);

    auto cstrs = [](const std::vector<std::string>& v) {
        std::vector<const char*> r; r.reserve(v.size());
        for (auto& s : v) r.push_back(s.c_str());
        return r;
    };

    std::cout << "Loading kernel_proj (CPU ONNX, runs at init only)..." << std::flush;
    Ort::Session kp_sess(env, cfg.kernel_proj_path.c_str(), cpu_opts);
    auto kp_in_names  = get_io_names(kp_sess, true);
    auto kp_out_names = get_io_names(kp_sess, false);
    auto kp_in_c  = cstrs(kp_in_names);
    auto kp_out_c = cstrs(kp_out_names);
    std::cout << " done (" << kp_in_names.size() << " in, " << kp_out_names.size() << " out)\n";

    std::cout << "Loading search_proj (CPU ONNX, every frame)..." << std::flush;
    Ort::Session sp_sess(env, cfg.search_proj_path.c_str(), cpu_opts);
    auto sp_in_names  = get_io_names(sp_sess, true);
    auto sp_out_names = get_io_names(sp_sess, false);
    auto sp_in_c  = cstrs(sp_in_names);
    auto sp_out_c = cstrs(sp_out_names);
    std::cout << " done (" << sp_in_names.size() << " in, " << sp_out_names.size() << " out)\n";

    std::cout << "Loading head (CPU ONNX, every frame)..." << std::flush;
    Ort::Session hd_sess(env, cfg.head_path.c_str(), cpu_opts);
    auto hd_in_names  = get_io_names(hd_sess, true);
    auto hd_out_names = get_io_names(hd_sess, false);
    auto hd_in_c  = cstrs(hd_in_names);
    auto hd_out_c = cstrs(hd_out_names);
    std::cout << " done (" << hd_in_names.size() << " in, " << hd_out_names.size() << " out)\n";

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ── Video I/O ────────────────────────────────────────────────────────────
    cv::VideoCapture cap(cfg.video_path);
    if (!cap.isOpened()) {
        fprintf(stderr, "Cannot open: %s\n", cfg.video_path.c_str()); return 1;
    }
    int VW       = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int VH       = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps_d = cap.get(cv::CAP_PROP_FPS);
    int total_f  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    printf("Video: %dx%d @%.2ffps, %d frames\n", VW, VH, fps_d, total_f);

    cv::VideoWriter writer;
    if (!cfg.output_path.empty()) {
        writer.open(cfg.output_path,
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps_d, cv::Size(VW, VH));
        if (!writer.isOpened()) {
            fprintf(stderr, "Cannot open output: %s\n", cfg.output_path.c_str()); return 1;
        }
    }

    // Display is implicitly on when doing interactive selection (window already open)
    if (cfg.bbox_w == 0 || cfg.bbox_h == 0) cfg.display = true;

    // ── Interactive bbox selection if not provided ───────────────────────────
    if (cfg.bbox_w == 0 || cfg.bbox_h == 0) {
        cv::Mat first;
        if (!cap.read(first)) {
            fprintf(stderr, "Cannot read first frame for ROI selection\n"); return 1;
        }
        printf("No --init_bbox given. Draw bounding box with mouse, then press ENTER or SPACE.\n"
               "Press C or ESC to cancel.\n");
        cv::Rect roi = cv::selectROI("Select target (ENTER to confirm, C to cancel)",
                                     first, false, false);
        cv::destroyWindow("Select target (ENTER to confirm, C to cancel)");
        if (roi.width == 0 || roi.height == 0) {
            fprintf(stderr, "No ROI selected — exiting.\n"); return 1;
        }
        cfg.bbox_x = roi.x;
        cfg.bbox_y = roi.y;
        cfg.bbox_w = roi.width;
        cfg.bbox_h = roi.height;
        printf("Selected bbox: x=%d y=%d w=%d h=%d\n",
               cfg.bbox_x, cfg.bbox_y, cfg.bbox_w, cfg.bbox_h);
        // Rewind so frame 0 is processed normally in the init block
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    // ── CSV output setup ─────────────────────────────────────────────────────
    FILE* csv_fp = nullptr;
    if (!cfg.score_csv_path.empty()) {
        csv_fp = fopen(cfg.score_csv_path.c_str(), "w");
        if (csv_fp) fprintf(csv_fp, "frame,score,cx,cy,w,h,reacq\n");
    }

    // ── Tracker state ────────────────────────────────────────────────────────
    double cx = cfg.bbox_x + (cfg.bbox_w - 1) / 2.0;
    double cy = cfg.bbox_y + (cfg.bbox_h - 1) / 2.0;
    float  tw = (float)cfg.bbox_w;
    float  th = (float)cfg.bbox_h;
    cv::Scalar avg_chans;
    std::vector<Tensor> zf;
    std::vector<Tensor> zf_views;  // kernel projections, cached across frames

    NumpyRNG nrng; nrng.seed(42);
    int N = cfg.n_particles;
    std::vector<Particle>  pf(N);
    std::vector<float>     wts(N, 1.0f / N);

    double t_pre = 0, t_srch = 0, t_sp = 0, t_xcorr = 0, t_head = 0, t_pf2 = 0;
    int    n_tracked = 0;
    auto   now_ms = []() {
        return (double)duration_cast<microseconds>(
            high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    };

    const int   REACQ_STREAK    = 100;
    const float REACQ_LOW       = 0.15f;
    const float REACQ_HIGH      = 0.90f;
    const int   REACQ_EXIT_FRAMES = 10;
    int   lo_streak = 0;
    int   hi_streak = 0;
    bool  in_reacq  = false;
    bool inited = false;
    int  fi = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (cfg.max_frames > 0 && fi >= cfg.max_frames) break;

        // ── Initialization ───────────────────────────────────────────────────
        if (!inited) {
            avg_chans = cv::mean(frame);

            float wz = tw + cfg.context_amount * (tw + th);
            float hz = th + cfg.context_amount * (tw + th);
            float sz = std::sqrt(wz * hz);

            auto tmpl_t = get_subwindow_tensor(frame, cx, cy,
                                                cfg.exemplar_size,
                                                std::round(sz), avg_chans);

            auto [zf_out, t_ms] = run_template_encoder(tmpl_enc, tmpl_t,
                                                         cfg.exemplar_size,
                                                         cfg.exemplar_size);
            zf = std::move(zf_out);
            printf("  template_encoder: %.1f ms  zf[0]=%ldx%ldx%ldx%ld\n",
                   t_ms,
                   (long)zf[0].shape[0], (long)zf[0].shape[1],
                   (long)zf[0].shape[2], (long)zf[0].shape[3]);

            // Kernel projection: zf_0,1,2 → view_3,8,...,view_28 (cached until template update)
            {
                std::vector<Ort::Value> kp_vals;
                for (int k = 0; k < 3; k++)
                    kp_vals.push_back(make_ort_value(mem, zf[k].data, zf[k].shape));
                auto kp_outs = run_session(kp_sess, kp_in_c, kp_vals, kp_out_c);
                zf_views.clear();
                for (auto& v : kp_outs) {
                    Tensor t = ort_to_tensor(v);
                    // kernel_proj outputs (C,1,kH,kW); memory layout == (1,C,kH,kW)
                    t.shape = {1, t.shape[0], t.shape[2], t.shape[3]};
                    zf_views.push_back(std::move(t));
                }
                printf("  kernel_proj: %zu views, each (1,256,5,5)\n", zf_views.size());
            }

            // Init particles
            float cx0 = cfg.bbox_x + cfg.bbox_w / 2.0f;
            float cy0 = cfg.bbox_y + cfg.bbox_h / 2.0f;
            std::vector<float> cx_init(N), cy_init(N);
            for (int i = 0; i < N; i++) cx_init[i] = (float)nrng.rk_gauss() * 2.0f;
            for (int i = 0; i < N; i++) cy_init[i] = (float)nrng.rk_gauss() * 2.0f;
            for (int i = 0; i < N; i++)
                pf[i] = { cx0 + cx_init[i], cy0 + cy_init[i], 0, 0, tw, th };
            wts.assign(N, 1.0f / N);

            cv::rectangle(frame, cv::Rect(cfg.bbox_x, cfg.bbox_y,
                                           cfg.bbox_w, cfg.bbox_h),
                          cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "f=0 init", cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 1);
            if (writer.isOpened()) writer.write(frame);
            if (cfg.display) {
                cv::imshow("SiamRPN++ Tracker  [Q/ESC = quit]", frame);
                cv::waitKey(1);
            }
            inited = true; fi++; continue;
        }

        // ── Per-frame tracking ───────────────────────────────────────────────
        float wz = tw + cfg.context_amount * (tw + th);
        float hz = th + cfg.context_amount * (tw + th);
        float sz = std::sqrt(wz * hz);
        float scale_z = (float)cfg.exemplar_size / sz;
        float s_x     = sz * ((float)cfg.instance_size / cfg.exemplar_size);

        // Full-frame reacquisition overrides search window center and scale
        float cx_use  = in_reacq ? (float)(VW / 2.0) : (float)cx;
        float cy_use  = in_reacq ? (float)(VH / 2.0) : (float)cy;
        float s_x_use = in_reacq ? (float)std::max(VW, VH) : s_x;

        // Preprocessing
        double tp = now_ms();
        auto srch_t = get_subwindow_tensor(frame, cx_use, cy_use,
                                            cfg.instance_size,
                                            std::round(s_x_use), avg_chans);
        t_pre += now_ms() - tp;

        // Search encoder (Metis)
        tp = now_ms();
        auto [xf_raw, srch_ms] = srch_enc.run(srch_t, 3,
                                                cfg.instance_size, cfg.instance_size);
        t_srch += now_ms() - tp;
        std::vector<Tensor>& xf = xf_raw;

        // ── Search projection (ORT, static weights) ──────────────────────────
        tp = now_ms();
        std::vector<Ort::Value> sp_vals;
        for (int k = 0; k < 3; k++)
            sp_vals.push_back(make_ort_value(mem, xf[k].data, xf[k].shape));
        auto sp_outs = run_session(sp_sess, sp_in_c, sp_vals, sp_out_c);
        t_sp += now_ms() - tp;

        // ── C++ AVX2 dw_xcorr (6 calls: 3 levels × cls+loc) ─────────────────
        tp = now_ms();
        std::vector<Tensor> xcorr_parts;
        xcorr_parts.reserve(6);
        for (int k = 0; k < 6; k++) {
            Tensor search_k = ort_to_tensor(sp_outs[k]);
            xcorr_parts.push_back(dw_xcorr(search_k, zf_views[k]));
        }
        t_xcorr += now_ms() - tp;

        // ── Head (ORT, static weights) ────────────────────────────────────────
        tp = now_ms();
        std::vector<Ort::Value> hd_vals;
        for (auto& xc : xcorr_parts)
            hd_vals.push_back(make_ort_value(mem, xc.data, xc.shape));
        auto xc_outs = run_session(hd_sess, hd_in_c, hd_vals, hd_out_c);
        t_head += now_ms() - tp;

        Tensor cls_t = ort_to_tensor(xc_outs[0]);
        Tensor loc_t = ort_to_tensor(xc_outs[1]);
        auto scores = decode_scores(cls_t.data.data(), cfg.anchor_num,
                                    cfg.score_size * cfg.score_size);
        auto bboxes = decode_bbox(loc_t.data.data(), total_anch, anchors);
        auto best   = select_best(scores, bboxes, cosine_win,
                                   scale_z, tw, th, cfg);

        // EMA size update (skipped during full-frame reacq — size estimate is unreliable)
        if (!in_reacq) {
            tw = tw * (1 - best.lr) + best.w * best.lr;
            th = th * (1 - best.lr) + best.h * best.lr;
            float clip_val = std::max((float)cfg.instance_size / scale_z,
                                      scale_z / (float)cfg.instance_size);
            tw = std::max(10.0f, std::min(clip_val, tw));
            th = std::max(10.0f, std::min(clip_val, th));
        }

        // ── Particle filter ──────────────────────────────────────────────────
        tp = now_ms();
        float psr = best.psr;
        auto& sm  = best.score_map_25x25;

        double cx_s = in_reacq ? VW / 2.0 : cx;
        double cy_s = in_reacq ? VH / 2.0 : cy;
        {
            std::vector<float> cx_n(N), cy_n(N), vx_n(N), vy_n(N), w_n(N), h_n(N);
            for (int i = 0; i < N; i++) cx_n[i] = (float)nrng.rk_gauss() * cfg.sigma_pos;
            for (int i = 0; i < N; i++) cy_n[i] = (float)nrng.rk_gauss() * cfg.sigma_pos;
            for (int i = 0; i < N; i++) vx_n[i] = (float)nrng.rk_gauss() * cfg.sigma_vel;
            for (int i = 0; i < N; i++) vy_n[i] = (float)nrng.rk_gauss() * cfg.sigma_vel;
            for (int i = 0; i < N; i++) w_n[i]  = (float)nrng.rk_gauss() * cfg.sigma_scale;
            for (int i = 0; i < N; i++) h_n[i]  = (float)nrng.rk_gauss() * cfg.sigma_scale;
            for (int i = 0; i < N; i++) {
                pf[i].cx += pf[i].vx + cx_n[i];
                pf[i].cy += pf[i].vy + cy_n[i];
                pf[i].vx += vx_n[i];
                pf[i].vy += vy_n[i];
                pf[i].vx = std::max(-cfg.max_vel, std::min(cfg.max_vel, pf[i].vx));
                pf[i].vy = std::max(-cfg.max_vel, std::min(cfg.max_vel, pf[i].vy));
                pf[i].w *= std::exp(w_n[i]);
                pf[i].h *= std::exp(h_n[i]);
                pf[i].cx = std::max(0.0f, std::min((float)VW, pf[i].cx));
                pf[i].cy = std::max(0.0f, std::min((float)VH, pf[i].cy));
            }
        }

        if (psr >= 2.0f) {
            float scale = 255.0f / (s_x_use + 1e-6f);
            std::vector<float> log_w(N);
            float max_lw = -1e30f;
            for (int p = 0; p < N; p++) {
                float sx_p = (float)((pf[p].cx - cx_s) * scale + 127.5);
                float sy_p = (float)((pf[p].cy - cy_s) * scale + 127.5);
                int fx = std::max(0, std::min(24, (int)((sx_p - 31.5f) / 8.0f)));
                int fy = std::max(0, std::min(24, (int)((sy_p - 31.5f) / 8.0f)));
                log_w[p] = sm[fy * cfg.score_size + fx] / cfg.tau;
                if (log_w[p] > max_lw) max_lw = log_w[p];
            }
            float wts_sum = 0;
            for (int p = 0; p < N; p++) {
                wts[p] = std::exp(log_w[p] - max_lw);
                wts_sum += wts[p];
            }
            for (auto& wv : wts) wv /= wts_sum + 1e-12f;
        }

        float ess = 0;
        for (float wv : wts) ess += wv*wv;
        ess = 1.0f / (ess + 1e-12f);
        if (ess < N / 2.0f) {
            auto idx = systematic_resample(wts, nrng);
            std::vector<Particle> new_pf(N);
            for (int p = 0; p < N; p++) new_pf[p] = pf[idx[p]];
            pf = new_pf;
            wts.assign(N, 1.0f / N);
            std::vector<float> cx_r(N), cy_r(N);
            for (int i = 0; i < N; i++) cx_r[i] = (float)nrng.rk_gauss() * cfg.roughen_std;
            for (int i = 0; i < N; i++) cy_r[i] = (float)nrng.rk_gauss() * cfg.roughen_std;
            for (int i = 0; i < N; i++) { pf[i].cx += cx_r[i]; pf[i].cy += cy_r[i]; }
        }

        double new_cx = 0.0, new_cy = 0.0;
        for (int p = 0; p < N; p++) {
            new_cx += (double)wts[p] * (double)pf[p].cx;
            new_cy += (double)wts[p] * (double)pf[p].cy;
        }
        cx = new_cx; cy = new_cy;

        // Full-frame reacquisition
        if (!in_reacq) {
            if (best.score < REACQ_LOW) {
                lo_streak++;
                if (lo_streak >= REACQ_STREAK) {
                    in_reacq = true;
                    lo_streak = 0;
                    for (int i = 0; i < N; i++) {
                        pf[i].cx = (float)(nrng.rk_double() * VW);
                        pf[i].cy = (float)(nrng.rk_double() * VH);
                        pf[i].vx = pf[i].vy = 0.0f;
                    }
                    wts.assign(N, 1.0f / N);
                    fprintf(stderr, "\n[REACQ ENTER] f=%d  score=%.3f  %d-frame streak"
                            "  switching to full-frame search\n",
                            fi, best.score, REACQ_STREAK);
                }
            } else {
                lo_streak = 0;
            }
        } else {
            if (best.score >= REACQ_HIGH) {
                hi_streak++;
                if (hi_streak >= REACQ_EXIT_FRAMES) {
                    in_reacq = false;
                    lo_streak = 0;
                    hi_streak = 0;
                    fprintf(stderr, "\n[REACQ EXIT]  f=%d  score=%.3f  %d consecutive high-score frames"
                            "  cx=%.1f cy=%.1f\n",
                            fi, best.score, REACQ_EXIT_FRAMES, cx, cy);
                }
            } else {
                hi_streak = 0;
            }
        }
        t_pf2 += now_ms() - tp;

        // ── Template update ──────────────────────────────────────────────────
        if (!in_reacq && fi % cfg.tmpl_freq == 0 && psr > cfg.tmpl_min_psr
            && tw < cfg.bbox_w*3 && th < cfg.bbox_h*3) {
            float wz2 = tw + cfg.context_amount * (tw + th);
            float hz2 = th + cfg.context_amount * (tw + th);
            float sz2 = std::sqrt(wz2 * hz2);
            auto zt = get_subwindow_tensor(frame, cx, cy,
                                            cfg.exemplar_size,
                                            std::round(sz2), avg_chans);
            auto [new_zf_raw, _] = run_template_encoder(tmpl_enc, zt,
                                                          cfg.exemplar_size,
                                                          cfg.exemplar_size);
            for (size_t k = 0; k < zf.size(); k++) {
                for (size_t p = 0; p < zf[k].data.size(); p++)
                    zf[k].data[p] = (1 - cfg.tmpl_alpha) * zf[k].data[p]
                                  + cfg.tmpl_alpha * new_zf_raw[k].data[p];
            }
            // Re-run kernel projection with updated zf
            std::vector<Ort::Value> kp_vals;
            for (int k = 0; k < 3; k++)
                kp_vals.push_back(make_ort_value(mem, zf[k].data, zf[k].shape));
            auto kp_outs = run_session(kp_sess, kp_in_c, kp_vals, kp_out_c);
            zf_views.clear();
            for (auto& v : kp_outs) {
                Tensor t = ort_to_tensor(v);
                t.shape = {1, t.shape[0], t.shape[2], t.shape[3]};
                zf_views.push_back(std::move(t));
            }
        }

        // ── Draw + write frame ───────────────────────────────────────────────
        int bx = (int)(cx - tw/2), by = (int)(cy - th/2);
        cv::Scalar box_color = in_reacq ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::rectangle(frame, cv::Rect(bx, by, (int)tw, (int)th), box_color, 2);
        cv::circle(frame, cv::Point((int)cx, (int)cy), 3, cv::Scalar(255, 0, 0), -1);
        if (in_reacq) {
            cv::putText(frame, "REACQUIRING", cv::Point(10, VH - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        if (csv_fp)
            fprintf(csv_fp, "%d,%.6f,%.3f,%.3f,%.3f,%.3f,%d\n",
                    fi, best.score, cx, cy, tw, th, (int)in_reacq);

        if (fi % 100 == 0) {
            fprintf(stderr, "  f=%5d  score=%.4f  cx=%.1f cy=%.1f  tw=%.1f th=%.1f"
                    "  srch=%.1fms%s  lo=%d hi=%d\n",
                    fi, best.score, cx, cy, tw, th, srch_ms,
                    in_reacq ? "  [REACQ]" : "", lo_streak, hi_streak);
        }

        if (writer.isOpened()) writer.write(frame);

        if (cfg.display) {
            cv::imshow("SiamRPN++ Tracker  [Q/ESC = quit]", frame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q' || key == 27)
                break;
        }

        n_tracked++; fi++;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    if (cfg.display) cv::destroyAllWindows();
    if (csv_fp) fclose(csv_fp);

    srch_enc.print_latency_breakdown("search_encoder");
    tmpl_enc.print_latency_breakdown("template_encoder");

    axr_destroy(AXR_OBJECT(axr_ctx));

    double tt = t_pre + t_srch + t_sp + t_xcorr + t_head + t_pf2;
    fprintf(stderr, "\n=== C++ (Metis, optimized) Timing over %d frames ===\n", n_tracked);
    fprintf(stderr, "Total tracked:              %.0f ms  →  %.2f fps\n",
            tt, n_tracked * 1000.0 / tt);
    fprintf(stderr, "preproc (OpenCV):           %.2f ms/f  (%.1f%%)\n",
            t_pre/n_tracked,    t_pre/tt*100);
    fprintf(stderr, "search_enc (Metis):         %.2f ms/f  (%.1f%%)\n",
            t_srch/n_tracked,   t_srch/tt*100);
    fprintf(stderr, "search_proj (ORT):          %.2f ms/f  (%.1f%%)\n",
            t_sp/n_tracked,     t_sp/tt*100);
    fprintf(stderr, "dw-xcorr (AVX2, 6×):        %.2f ms/f  (%.1f%%)\n",
            t_xcorr/n_tracked,  t_xcorr/tt*100);
    fprintf(stderr, "head (ORT):                 %.2f ms/f  (%.1f%%)\n",
            t_head/n_tracked,   t_head/tt*100);
    fprintf(stderr, "particle_filter:            %.2f ms/f  (%.1f%%)\n",
            t_pf2/n_tracked,    t_pf2/tt*100);

    return 0;
}
