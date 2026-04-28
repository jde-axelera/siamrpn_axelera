// poc_siamrpn.cpp — SiamRPN++ LT tracker, exact C++ port of poc_siamrpn.py
//
// Pipeline per frame:
//   [CPU ORT]  template_encoder_r50lt.onnx → 6 kernel tensors  (once at init)
//   [Metis]    siamrpn++onnx_255/1/model.json → 6 search features (per frame)
//   [CPU]      depthwise XCorr             → xcorr_features (1,1792,25,25)
//   [CPU ORT]  siamrpn_head_dyn.onnx       → cls (1,10,25,25), loc (1,20,25,25)
//
// Default: instance_size=255, score_size=25 (~17ms Metis encoder).
// For 351px/37-score:  --instance_size 351 --head siamrpn_head_37.onnx
//                      --search_encoder build/siamrpn++onnx/siamrpn++onnx/1/model.json
//
// Long-term mode: 2×2 tiled 255px search on 510px window, win_inf≈0.
//
// Usage:
//   ./poc_siamrpn_cpp \
//     --search_encoder build/siamrpn++onnx_255/siamrpn++onnx_255/1/model.json \
//     --template_encoder template_encoder_r50lt.onnx \
//     --head             siamrpn_head_dyn.onnx \
//     --video            coyote.mp4 \
//     --output           out.mp4 \
//     [--init_bbox 695,345,20,30]  (omit → mouse select)
//     [--display]
//     [--max_frames 2001]
//     [--instance_size 255]        (255 default, or 351 for old encoder)
//     [--lost_instance_size 510]   (default 2×instance_size)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <axruntime/axruntime.h>

using namespace std::chrono;

// ─── Config (mirrors Python Config exactly) ───────────────────────────────────
struct Config {
    int   exemplar_size      = 127;
    int   instance_size      = 255;
    int   lost_instance_size = -1;   // -1 → auto: 2 × instance_size
    int   num_tiles          = 2;

    int effective_lost_size() const {
        return lost_instance_size > 0 ? lost_instance_size : instance_size * 2;
    }

    float penalty_k          = 0.05f;
    float window_influence   = 0.35f;
    float lr                 = 0.28f;
    float conf_low           = 0.8f;
    float conf_high          = 0.985f;
    bool  enable_lt_mode     = true;

    int   anchor_stride      = 8;
    float anchor_scale       = 8.0f;
    int   base_size          = 8;
    float context_amount     = 0.5f;
    std::vector<float> anchor_ratios = {0.33f, 0.5f, 1.0f, 2.0f, 3.0f};

    int   anchor_num()  const { return (int)anchor_ratios.size(); }
    int   score_size()  const {
        return (instance_size - exemplar_size) / anchor_stride + 1 + base_size;
    }

    std::string srch_enc_path;
    std::string tmpl_enc_path;
    std::string head_path;
    std::string video_path;
    std::string output_path;

    int  aipu_cores = 4;
    bool display    = false;
    int  bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
    int  max_frames = -1;

    bool  use_pf  = true;
    int   pf_n    = 200;
    float pf_std  = 12.0f;   // motion noise std (pixels) per frame

    // Global scan: when stuck in LT mode for this many frames below stuck_thresh,
    // search all 4 frame quadrants and jump to the best one (0 = disabled).
    int   lt_stuck_timeout   = 30;
    float lt_stuck_threshold = 0.3f;
};

// ─── Tensor ──────────────────────────────────────────────────────────────────
struct Tensor {
    std::vector<float>   data;
    std::vector<int64_t> shape;
    size_t numel() const {
        size_t n = 1; for (auto s : shape) n *= (size_t)s; return n;
    }
};

// ─── Anchor generation (Python _generate_anchors exact) ──────────────────────
// Returns (anchor_num * score_size * score_size, 4) in [x1,y1,x2,y2] format.
std::vector<float> generate_anchors(const Config& cfg, int ss) {
    int N  = cfg.anchor_num();
    int SS = ss * ss;
    std::vector<float> out(N * SS * 4);
    int ori = -(ss / 2) * cfg.anchor_stride;

    int k = 0;
    for (float r : cfg.anchor_ratios) {
        int ws = (int)std::sqrt((float)(cfg.anchor_stride * cfg.anchor_stride) / r);
        int hs = (int)(ws * r);
        float aw = ws * cfg.anchor_scale;
        float ah = hs * cfg.anchor_scale;
        for (int i = 0; i < ss; i++) {
            for (int j = 0; j < ss; j++) {
                float x = ori + j * cfg.anchor_stride;
                float y = ori + i * cfg.anchor_stride;
                int idx = k * SS + i * ss + j;
                out[idx*4+0] = x - aw/2;
                out[idx*4+1] = y - ah/2;
                out[idx*4+2] = x + aw/2;
                out[idx*4+3] = y + ah/2;
            }
        }
        k++;
    }
    return out;
}

// ─── Cosine window ────────────────────────────────────────────────────────────
std::vector<float> make_cosine_window(int anchor_num, int ss) {
    std::vector<float> h(ss);
    for (int i = 0; i < ss; i++)
        h[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (ss - 1));
    std::vector<float> win(anchor_num * ss * ss);
    for (int k = 0; k < anchor_num; k++)
        for (int i = 0; i < ss; i++)
            for (int j = 0; j < ss; j++)
                win[k*ss*ss + i*ss + j] = h[i] * h[j];
    return win;
}

// ─── Subwindow (Python _get_subwindow) ───────────────────────────────────────
// Returns HWC uint8 crop, model_sz×model_sz.
cv::Mat get_subwindow(const cv::Mat& img, double cx, double cy,
                      int model_sz, double orig_sz, const cv::Scalar& avg) {
    double c = (orig_sz + 1) / 2.0;
    int xmin = (int)(cx - c), xmax = xmin + (int)orig_sz;
    int ymin = (int)(cy - c), ymax = ymin + (int)orig_sz;

    int lp = std::max(0, -xmin), tp = std::max(0, -ymin);
    int rp = std::max(0, xmax - img.cols), bp = std::max(0, ymax - img.rows);
    xmin += lp; xmax += lp; ymin += tp; ymax += tp;

    cv::Mat padded;
    if (lp || tp || rp || bp)
        cv::copyMakeBorder(img, padded, tp, bp, lp, rp, cv::BORDER_CONSTANT, avg);
    else
        padded = img;

    cv::Mat crop = padded(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)).clone();
    if (model_sz != (int)orig_sz)
        cv::resize(crop, crop, cv::Size(model_sz, model_sz));
    return crop;
}

// HWC uint8 → float32 NCHW
std::vector<float> to_nchw(const cv::Mat& m) {
    int H = m.rows, W = m.cols;
    std::vector<float> t(3 * H * W);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                t[c*H*W + i*W + j] = (float)m.at<cv::Vec3b>(i,j)[c];
    return t;
}

// ─── Depthwise XCorr ─────────────────────────────────────────────────────────
// search: (1,C,H,W), kernel: (1,C,kH,kW) → output: (1,C,H-kH+1,W-kW+1)
//
// Formulation: for each kernel offset (dr,dc), accumulate kval[c] * X[c,r+dr,:+dc]
// into out[c,r,:].  The innermost loop over oW is unit-stride in both X and out,
// so GCC -O3 -mavx2 -mfma auto-vectorizes it to 8-wide FMAs.  Outer loop order
// (c → dr,dc → r) keeps X[c] (6.7 KB) and out[c] (5.4 KB) in L1 cache for all
// 25 kernel positions, avoiding the 240 MB im2col write that BLAS would require.
Tensor dw_xcorr(const Tensor& search, const Tensor& kernel) {
    const int C  = (int)search.shape[1];
    const int H  = (int)search.shape[2], W  = (int)search.shape[3];
    const int kH = (int)kernel.shape[2], kW = (int)kernel.shape[3];
    const int oH = H - kH + 1, oW = W - kW + 1;

    Tensor out;
    out.shape = {1, (int64_t)C, (int64_t)oH, (int64_t)oW};
    out.data.assign((size_t)C * oH * oW, 0.0f);

    // __restrict__ tells GCC that X, K, O are non-overlapping; without it the
    // inner loop only gets a scalar fallback due to conservative aliasing.
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
                    // ivdep asserts no loop-carried dependencies → 8-wide AVX2 FMA
                    #pragma GCC ivdep
                    for (int oc = 0; oc < oW; ++oc)
                        d[oc] += kval * s[oc];
                }
            }
        }
    }
    return out;
}

// Concatenate 6 xcorr results along channel dim → (1, total_C, H, W)
Tensor concat_xcorr(const std::vector<Tensor>& parts) {
    int64_t H = parts[0].shape[2], W = parts[0].shape[3];
    int64_t total_C = 0;
    for (auto& p : parts) total_C += p.shape[1];

    Tensor out;
    out.shape = {1, total_C, H, W};
    out.data.resize((size_t)(total_C * H * W));

    int64_t off = 0;
    for (auto& p : parts) {
        int64_t C = p.shape[1];
        memcpy(out.data.data() + off * H * W,
               p.data.data(),
               (size_t)(C * H * W) * sizeof(float));
        off += C;
    }
    return out;
}

// ─── Axelera Metis encoder ────────────────────────────────────────────────────
struct AxeleraEncoder {
    axrContext*       ctx  = nullptr;
    axrModel*         mdl  = nullptr;
    axrConnection*    conn = nullptr;
    axrModelInstance* inst = nullptr;

    std::vector<axrTensorInfo>       in_infos,  out_infos;
    std::vector<std::vector<int8_t>> in_bufs,   out_bufs;

    void init(axrContext* c, const char* path, int aipu_cores) {
        ctx = c;
        mdl = axr_load_model(ctx, path);
        if (!mdl) {
            fprintf(stderr, "axr_load_model failed: %s\n  error: %s\n",
                    path, axr_last_error_string(AXR_OBJECT(ctx)));
            exit(1);
        }
        conn = axr_device_connect(ctx, nullptr, 1, nullptr);
        if (!conn) {
            fprintf(stderr, "axr_device_connect failed: %s\n",
                    axr_last_error_string(AXR_OBJECT(ctx)));
            exit(1);
        }
        char props[64];
        snprintf(props, sizeof(props), "aipu_cores=%d\nnum_sub_devices=1", aipu_cores);
        axrProperties* p = axr_create_properties(ctx, props);
        inst = axr_load_model_instance(conn, mdl, p);
        axr_destroy(AXR_OBJECT(p));
        if (!inst) {
            fprintf(stderr, "axr_load_model_instance failed: %s\n",
                    axr_last_error_string(AXR_OBJECT(ctx)));
            exit(1);
        }
        for (size_t i = 0; i < axr_num_model_inputs(mdl); i++) {
            auto info = axr_get_model_input(mdl, i);
            in_infos.push_back(info);
            in_bufs.emplace_back(axr_tensor_size(&info), (int8_t)info.zero_point);
        }
        for (size_t i = 0; i < axr_num_model_outputs(mdl); i++) {
            auto info = axr_get_model_output(mdl, i);
            out_infos.push_back(info);
            out_bufs.emplace_back(axr_tensor_size(&info), 0);
        }
    }

    void quantize(const float* nchw, int C, int H, int W,
                  std::vector<int8_t>& buf, const axrTensorInfo& info) {
        int pH = (int)info.dims[1], pW = (int)info.dims[2], pC = (int)info.dims[3];
        int h0 = (int)info.padding[1][0], w0 = (int)info.padding[2][0];
        int c0 = (int)info.padding[3][0];
        float sc = (float)info.scale;
        int   zp = info.zero_point;
        std::fill(buf.begin(), buf.end(), (int8_t)zp);
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++) {
                    float q = std::round(nchw[c*H*W + h*W + w] / sc + zp);
                    q = std::max(-128.0f, std::min(127.0f, q));
                    buf[(h+h0)*pW*pC + (w+w0)*pC + (c+c0)] = (int8_t)(int)q;
                }
    }

    Tensor dequantize(const std::vector<int8_t>& buf, const axrTensorInfo& info) {
        int pH = (int)info.dims[1], pW = (int)info.dims[2], pC = (int)info.dims[3];
        int h0 = (int)info.padding[1][0], h1 = (int)info.padding[1][1];
        int w0 = (int)info.padding[2][0], w1 = (int)info.padding[2][1];
        int c0 = (int)info.padding[3][0], c1 = (int)info.padding[3][1];
        int H = pH-h0-h1, W = pW-w0-w1, C = pC-c0-c1;
        float sc = (float)info.scale;
        int   zp = info.zero_point;
        Tensor t;
        t.shape = {1, (int64_t)C, (int64_t)H, (int64_t)W};
        t.data.resize(C * H * W);
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++)
                    t.data[c*H*W + h*W + w] =
                        ((float)(int)buf[(h+h0)*pW*pC + (w+w0)*pC + (c+c0)] - zp) * sc;
        return t;
    }

    double t_run = 0;
    long   n_runs = 0;

    std::pair<std::vector<Tensor>, double> run(const std::vector<float>& nchw,
                                                int C, int H, int W) {
        auto ms = [](auto t0) {
            return duration_cast<microseconds>(
                high_resolution_clock::now() - t0).count() / 1000.0;
        };
        auto t0 = high_resolution_clock::now();
        quantize(nchw.data(), C, H, W, in_bufs[0], in_infos[0]);

        std::vector<axrArgument> ia(in_bufs.size()), oa(out_bufs.size());
        for (size_t i = 0; i < in_bufs.size(); i++)
            ia[i] = {in_bufs[i].data(), -1, 0, in_bufs[i].size()};
        for (size_t i = 0; i < out_bufs.size(); i++)
            oa[i] = {out_bufs[i].data(), -1, 0, out_bufs[i].size()};

        axrResult r = axr_run_model_instance(inst, ia.data(), ia.size(),
                                              oa.data(), oa.size());
        if (r != AXR_SUCCESS) {
            fprintf(stderr, "axr_run_model_instance: %s\n", axr_error_string(r));
            exit(1);
        }
        double elapsed = ms(t0);
        t_run += elapsed; n_runs++;

        std::vector<Tensor> outs;
        outs.reserve(out_bufs.size());
        for (size_t i = 0; i < out_bufs.size(); i++)
            outs.push_back(dequantize(out_bufs[i], out_infos[i]));
        return {std::move(outs), elapsed};
    }
};

// ─── ORT helpers ─────────────────────────────────────────────────────────────
static std::vector<std::string> ort_names(Ort::Session& s, bool input) {
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n = input ? s.GetInputCount() : s.GetOutputCount();
    std::vector<std::string> v(n);
    for (size_t i = 0; i < n; i++) {
        auto p = input ? s.GetInputNameAllocated(i, alloc)
                       : s.GetOutputNameAllocated(i, alloc);
        v[i] = p.get();
    }
    return v;
}
static std::vector<const char*> cstrs(const std::vector<std::string>& v) {
    std::vector<const char*> r; r.reserve(v.size());
    for (auto& s : v) r.push_back(s.c_str());
    return r;
}
static Tensor ort_to_tensor(Ort::Value& v) {
    auto info = v.GetTensorTypeAndShapeInfo();
    Tensor t;
    t.shape = info.GetShape();
    const float* p = v.GetTensorData<float>();
    t.data.assign(p, p + info.GetElementCount());
    return t;
}

// ─── Score/bbox decoding (Python _convert_score / _convert_bbox) ─────────────
// cls: (1, 2*anchor_num, ss, ss) → softmax scores, shape (N,)
std::vector<float> decode_scores(const float* cls, int anchor_num, int SS) {
    int N = anchor_num * SS;
    std::vector<float> s(N);
    for (int i = 0; i < N; i++) {
        float s0 = cls[i], s1 = cls[N + i];
        float m = std::max(s0, s1);
        float e0 = std::exp(s0 - m), e1 = std::exp(s1 - m);
        s[i] = e1 / (e0 + e1);
    }
    return s;
}

// loc: (1,4*anchor_num,ss,ss), anchors: (N,4) [x1,y1,x2,y2]
// Returns N bboxes as (cx,cy,w,h) in search-crop space.
struct BBox { float cx, cy, w, h; };
std::vector<BBox> decode_bboxes(const float* loc, int N,
                                 const std::vector<float>& anchors) {
    std::vector<BBox> b(N);
    for (int i = 0; i < N; i++) {
        float aw = anchors[i*4+2] - anchors[i*4+0];
        float ah = anchors[i*4+3] - anchors[i*4+1];
        float acx = (anchors[i*4+0] + anchors[i*4+2]) / 2;
        float acy = (anchors[i*4+1] + anchors[i*4+3]) / 2;
        b[i] = { loc[0*N+i]*aw + acx, loc[1*N+i]*ah + acy,
                 std::exp(loc[2*N+i])*aw, std::exp(loc[3*N+i])*ah };
    }
    return b;
}

// ─── Penalty + best selection (Python track() penalty block) ─────────────────
static float sz_wh(float w, float h) {
    float p = (w + h) * 0.5f;
    return std::sqrt((w + p) * (h + p));
}

struct Best { int idx; float score, lr; BBox bbox; };

Best select_best(const std::vector<float>& scores, const std::vector<BBox>& bboxes,
                 const std::vector<float>& window, float win_inf,
                 float scale_z, float prev_w, float prev_h,
                 float penalty_k, float lr_cfg) {
    int N = (int)scores.size();
    float psz = sz_wh(prev_w * scale_z, prev_h * scale_z);
    float pr  = prev_w / (prev_h + 1e-6f);

    std::vector<float> pscore(N), penalty(N);
    for (int i = 0; i < N; i++) {
        float sc = std::max(sz_wh(bboxes[i].w, bboxes[i].h) / (psz + 1e-6f),
                            psz / (sz_wh(bboxes[i].w, bboxes[i].h) + 1e-6f));
        float ar = bboxes[i].w / (bboxes[i].h + 1e-6f);
        float rc = std::max(ar / (pr + 1e-6f), pr / (ar + 1e-6f));
        penalty[i] = std::exp(-(sc * rc - 1.0f) * penalty_k);
        pscore[i]  = penalty[i] * scores[i] * (1 - win_inf) + window[i] * win_inf;
    }
    int best = (int)(std::max_element(pscore.begin(), pscore.end()) - pscore.begin());
    float best_score = scores[best];
    float best_lr    = penalty[best] * best_score * lr_cfg;
    return { best, best_score, best_lr,
             { bboxes[best].cx / scale_z, bboxes[best].cy / scale_z,
               bboxes[best].w  / scale_z, bboxes[best].h  / scale_z } };
}

// ─── Particle Filter ─────────────────────────────────────────────────────────
// Runs in normal mode only. Particles live in image coordinates.
// Predict: add Gaussian motion noise. Update: weight by max score at each
// particle's cell in the score grid. Estimate: weighted mean → new center.
struct PF {
    std::vector<float> px, py, w;
    int n = 0;

    void init(float cx, float cy, int n_, float spread, std::mt19937& rng) {
        n = n_;
        px.resize(n); py.resize(n); w.assign(n, 1.f / n);
        std::normal_distribution<float> d(0.f, spread);
        for (int i = 0; i < n; i++) { px[i] = cx + d(rng); py[i] = cy + d(rng); }
    }

    void predict(float std_px, std::mt19937& rng) {
        std::normal_distribution<float> d(0.f, std_px);
        for (int i = 0; i < n; i++) { px[i] += d(rng); py[i] += d(rng); }
    }

    // Build ss×ss max-score grid. scores layout: scores[a*ss*ss + r*ss + c].
    static std::vector<float> build_grid(const std::vector<float>& scores, int AN, int ss) {
        std::vector<float> g(ss * ss, 0.f);
        for (int a = 0; a < AN; a++)
            for (int p = 0; p < ss * ss; p++)
                g[p] = std::max(g[p], scores[a * ss * ss + p]);
        return g;
    }

    // Weight particles by score at their image position in the score grid.
    // Score grid is centred on (img_cx, img_cy) with spacing stride/scale_z.
    void update(const std::vector<float>& grid, float img_cx, float img_cy,
                float scale_z, int stride, int ss) {
        float wsum = 0.f;
        for (int i = 0; i < n; i++) {
            float gc = (px[i] - img_cx) * scale_z / stride + ss * 0.5f;
            float gr = (py[i] - img_cy) * scale_z / stride + ss * 0.5f;
            int c = std::max(0, std::min(ss - 1, (int)std::round(gc)));
            int r = std::max(0, std::min(ss - 1, (int)std::round(gr)));
            w[i] = grid[r * ss + c];
            wsum += w[i];
        }
        if (wsum < 1e-6f) { std::fill(w.begin(), w.end(), 1.f / n); return; }
        for (auto& wi : w) wi /= wsum;
    }

    void resample(std::mt19937& rng) {
        std::vector<float> cum(n);
        cum[0] = w[0];
        for (int i = 1; i < n; i++) cum[i] = cum[i-1] + w[i];
        std::vector<float> npx(n), npy(n);
        std::uniform_real_distribution<float> u(0.f, 1.f / n);
        float r = u(rng);
        int j = 0;
        for (int i = 0; i < n; i++) {
            float T = r + (float)i / n;
            while (j < n - 1 && cum[j] < T) j++;
            npx[i] = px[j]; npy[i] = py[j];
        }
        px = npx; py = npy;
        std::fill(w.begin(), w.end(), 1.f / n);
    }

    std::pair<float,float> estimate() const {
        float ex = 0.f, ey = 0.f;
        for (int i = 0; i < n; i++) { ex += w[i] * px[i]; ey += w[i] * py[i]; }
        return {ex, ey};
    }

    void recenter(float cx, float cy) {
        for (int i = 0; i < n; i++) { px[i] = cx; py[i] = cy; }
        std::fill(w.begin(), w.end(), 1.f / n);
    }
};

// Thread-local timing accumulators for xcorr and head (summed at end of main).
thread_local double tl_t_xcorr_only = 0;
thread_local double tl_t_head_only  = 0;

// ─── Run xcorr + head for one set of search features ─────────────────────────
// Applies tile offset correction to bboxes before returning.
std::pair<std::vector<float>, std::vector<BBox>>
run_xcorr_head(const std::vector<Tensor>& xf,
               const std::vector<Tensor>& zf,
               const std::vector<float>& anchors, int anchor_num, int ss,
               float tile_ox, float tile_oy, float tile_sz, float full_sz,
               Ort::Session& head_sess,
               const std::vector<const char*>& in_c,
               const std::vector<const char*>& out_c,
               Ort::MemoryInfo& mem) {
    auto clk = []{ return duration_cast<microseconds>(
        high_resolution_clock::now().time_since_epoch()).count() / 1000.0; };

    double t0 = clk();
    std::vector<Tensor> parts;
    parts.reserve(6);
    for (int i = 0; i < 6; i++)
        parts.push_back(dw_xcorr(xf[i], zf[i]));
    Tensor xcorr = concat_xcorr(parts);
    tl_t_xcorr_only += clk() - t0;

    t0 = clk();
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, xcorr.data.data(), xcorr.data.size(),
        xcorr.shape.data(), xcorr.shape.size()));
    auto outs = head_sess.Run(Ort::RunOptions{nullptr},
                              in_c.data(), inputs.data(), 1,
                              out_c.data(), out_c.size());
    tl_t_head_only += clk() - t0;

    Tensor cls_t = ort_to_tensor(outs[0]);
    Tensor loc_t = ort_to_tensor(outs[1]);
    int N = anchor_num * ss * ss;
    auto scores = decode_scores(cls_t.data.data(), anchor_num, ss * ss);
    auto bboxes = decode_bboxes(loc_t.data.data(), N, anchors);

    // Tile offset correction (Python: bbox[0] += tx + tile_size/2 - full_size/2)
    for (auto& b : bboxes) {
        b.cx += tile_ox + tile_sz / 2 - full_sz / 2;
        b.cy += tile_oy + tile_sz / 2 - full_sz / 2;
    }
    return {scores, bboxes};
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--search_encoder"      && i+1 < argc) cfg.srch_enc_path      = argv[++i];
        else if (a == "--template_encoder"    && i+1 < argc) cfg.tmpl_enc_path      = argv[++i];
        else if (a == "--head"                && i+1 < argc) cfg.head_path          = argv[++i];
        else if (a == "--video"               && i+1 < argc) cfg.video_path         = argv[++i];
        else if (a == "--output"              && i+1 < argc) cfg.output_path        = argv[++i];
        else if (a == "--aipu_cores"          && i+1 < argc) cfg.aipu_cores         = std::stoi(argv[++i]);
        else if (a == "--max_frames"          && i+1 < argc) cfg.max_frames         = std::stoi(argv[++i]);
        else if (a == "--instance_size"       && i+1 < argc) cfg.instance_size      = std::stoi(argv[++i]);
        else if (a == "--lost_instance_size"  && i+1 < argc) cfg.lost_instance_size = std::stoi(argv[++i]);
        else if (a == "--display")                           cfg.display            = true;
        else if (a == "--no_pf")                             cfg.use_pf             = false;
        else if (a == "--pf_n"                && i+1 < argc) cfg.pf_n              = std::stoi(argv[++i]);
        else if (a == "--pf_std"              && i+1 < argc) cfg.pf_std            = std::stof(argv[++i]);
        else if (a == "--lt_stuck_timeout"    && i+1 < argc) cfg.lt_stuck_timeout  = std::stoi(argv[++i]);
        else if (a == "--lt_stuck_thresh"     && i+1 < argc) cfg.lt_stuck_threshold= std::stof(argv[++i]);
        else if (a == "--init_bbox"           && i+1 < argc)
            sscanf(argv[++i], "%d,%d,%d,%d",
                   &cfg.bbox_x, &cfg.bbox_y, &cfg.bbox_w, &cfg.bbox_h);
    }

    const int ss  = cfg.score_size();   // 25 for 255px, 37 for 351px
    const int AN  = cfg.anchor_num();   // 5
    const int N   = AN * ss * ss;       // 3125 for 255px, 6845 for 351px

    printf("Config: instance_size=%d  score_size=%d  lost_instance_size=%d\n",
           cfg.instance_size, ss, cfg.effective_lost_size());

    auto anchors     = generate_anchors(cfg, ss);
    auto cosine_win  = make_cosine_window(AN, ss);

    // ── Load Axelera search encoder ─────────────────────────────────────────
    axrContext* axr_ctx = axr_create_context();
    if (!axr_ctx) { fprintf(stderr, "axr_create_context failed\n"); return 1; }

    // Normal-mode encoder: uses all cores for lowest latency on a single tile.
    AxeleraEncoder srch_enc;
    printf("Loading search encoder (Metis)...\n"); fflush(stdout);
    srch_enc.init(axr_ctx, cfg.srch_enc_path.c_str(), cfg.aipu_cores);
    printf("  done — %zu outputs\n", srch_enc.out_infos.size());

    // LT tile workers: 3 workers × 1 AIPU core = 3 connections.
    // 4 LT tiles are distributed round-robin (worker i gets tiles where t%3==i),
    // so worker 0 handles tiles {0,3} sequentially and workers 1,2 each handle 1.
    // Together with srch_enc (1 connection), this uses all 4 available sub-devices.
    const int N_TILE_WORKERS = 3;
    std::vector<AxeleraEncoder> tile_encs(N_TILE_WORKERS);
    printf("Loading %d tile encoders (Metis, 1 core each)...\n", N_TILE_WORKERS);
    fflush(stdout);
    for (int i = 0; i < N_TILE_WORKERS; i++)
        tile_encs[i].init(axr_ctx, cfg.srch_enc_path.c_str(), 1);
    printf("  done\n");

    // ── Load template encoder (CPU ORT) ────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "poc_siamrpn");
    Ort::SessionOptions cpu_opts;
    cpu_opts.SetIntraOpNumThreads(4);

    printf("Loading template encoder (CPU ORT)...\n"); fflush(stdout);
    Ort::Session tmpl_sess(env, cfg.tmpl_enc_path.c_str(), cpu_opts);
    auto tmpl_out_names = ort_names(tmpl_sess, false);
    auto tmpl_out_c     = cstrs(tmpl_out_names);
    auto tmpl_in_names  = ort_names(tmpl_sess, true);
    auto tmpl_in_c      = cstrs(tmpl_in_names);
    printf("  done — %zu kernel outputs\n", tmpl_out_names.size());

    // ── Load head (CPU ORT) — one session per tile worker for thread safety ──
    printf("Loading head (CPU ORT, %d sessions)...\n", 1 + N_TILE_WORKERS);
    fflush(stdout);
    Ort::Session head_sess(env, cfg.head_path.c_str(), cpu_opts);
    auto head_in_names  = ort_names(head_sess, true);
    auto head_out_names = ort_names(head_sess, false);
    auto head_in_c      = cstrs(head_in_names);
    auto head_out_c     = cstrs(head_out_names);
    // Extra sessions for parallel tile inference
    std::vector<Ort::Session> tile_head_sess;
    tile_head_sess.reserve(N_TILE_WORKERS);
    for (int i = 0; i < N_TILE_WORKERS; i++)
        tile_head_sess.emplace_back(env, cfg.head_path.c_str(), cpu_opts);
    printf("  done\n");

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ── Video I/O ───────────────────────────────────────────────────────────
    cv::VideoCapture cap(cfg.video_path);
    if (!cap.isOpened()) {
        fprintf(stderr, "Cannot open: %s\n", cfg.video_path.c_str()); return 1;
    }
    int VW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int VH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (cfg.max_frames > 0) total = std::min(total, cfg.max_frames);
    printf("Video: %dx%d @%.2ffps, processing %d frames\n", VW, VH, fps, total);

    cv::VideoWriter writer;
    if (!cfg.output_path.empty()) {
        writer.open(cfg.output_path,
                    cv::VideoWriter::fourcc('m','p','4','v'), fps, {VW, VH});
        if (!writer.isOpened()) {
            fprintf(stderr, "Cannot open output: %s\n", cfg.output_path.c_str()); return 1;
        }
    }

    // ── Interactive bbox selection ──────────────────────────────────────────
    if (cfg.bbox_w == 0 || cfg.bbox_h == 0) cfg.display = true;
    if (cfg.bbox_w == 0 || cfg.bbox_h == 0) {
        cv::Mat first; cap.read(first);
        printf("Draw bounding box, press ENTER/SPACE to confirm.\n");
        cv::Rect roi = cv::selectROI("Select target", first, false, false);
        cv::destroyWindow("Select target");
        if (roi.width == 0 || roi.height == 0) { fprintf(stderr, "No ROI.\n"); return 1; }
        cfg.bbox_x = roi.x; cfg.bbox_y = roi.y;
        cfg.bbox_w = roi.width; cfg.bbox_h = roi.height;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    // ── Tracker state ───────────────────────────────────────────────────────
    double cx = cfg.bbox_x + cfg.bbox_w / 2.0;
    double cy = cfg.bbox_y + cfg.bbox_h / 2.0;
    float  tw = (float)cfg.bbox_w, th = (float)cfg.bbox_h;
    cv::Scalar avg_chans;
    std::vector<Tensor> zf;           // template kernels (set at init)
    bool inited = false;
    bool longterm = false;
    PF pf;
    std::mt19937 pf_rng(42);
    int lt_stuck_frames = 0;

    auto now_ms = []() {
        return duration_cast<microseconds>(
            high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    };

    double t_backbone = 0, t_xcorr = 0, t_head = 0, t_pre = 0;
    int fi = 0;
    double t_wall_start = now_ms();
    cv::Mat frame;

    while (cap.read(frame)) {
        if (fi >= total) break;

        // ── Init (frame 0) ──────────────────────────────────────────────────
        if (!inited) {
            avg_chans = cv::mean(frame);
            float wz = tw + cfg.context_amount * (tw + th);
            float hz = th + cfg.context_amount * (tw + th);
            float sz = std::sqrt(wz * hz);

            auto tmpl_crop = get_subwindow(frame, cx, cy,
                                           cfg.exemplar_size, std::round(sz), avg_chans);
            auto tmpl_t = to_nchw(tmpl_crop);

            // Run template encoder (CPU ORT)
            std::vector<int64_t> tmpl_shape = {1, 3, cfg.exemplar_size, cfg.exemplar_size};
            std::vector<Ort::Value> tin;
            tin.push_back(Ort::Value::CreateTensor<float>(
                mem, tmpl_t.data(), tmpl_t.size(),
                tmpl_shape.data(), tmpl_shape.size()));
            auto t0 = high_resolution_clock::now();
            auto tout = tmpl_sess.Run(Ort::RunOptions{nullptr},
                                      tmpl_in_c.data(), tin.data(), 1,
                                      tmpl_out_c.data(), tmpl_out_c.size());
            double te_ms = duration_cast<microseconds>(
                high_resolution_clock::now() - t0).count() / 1000.0;

            zf.clear();
            for (auto& v : tout) zf.push_back(ort_to_tensor(v));
            printf("  template_encoder (CPU): %.1fms  k0:%ldx%ldx%ldx%ld\n",
                   te_ms, (long)zf[0].shape[0], (long)zf[0].shape[1],
                   (long)zf[0].shape[2], (long)zf[0].shape[3]);

            // Draw init bbox
            cv::rectangle(frame, {cfg.bbox_x, cfg.bbox_y, cfg.bbox_w, cfg.bbox_h},
                          {0, 255, 0}, 2);
            cv::putText(frame, "f=0 INIT", {10, 25},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 1);
            if (writer.isOpened()) writer.write(frame);
            if (cfg.display) { cv::imshow("poc_siamrpn++ [Q/ESC=quit]", frame); cv::waitKey(1); }
            if (cfg.use_pf)
                pf.init((float)cx, (float)cy, cfg.pf_n, 5.0f, pf_rng);
            inited = true; fi++; continue;
        }

        // ── Per-frame tracking ──────────────────────────────────────────────
        float wz = tw + cfg.context_amount * (tw + th);
        float hz = th + cfg.context_amount * (tw + th);
        float sz = std::sqrt(wz * hz);
        float scale_z = (float)cfg.exemplar_size / sz;

        std::vector<float> all_scores;
        std::vector<BBox>  all_bboxes;
        float scale_z_use, win_inf;

        if (!longterm) {
            // ── Normal mode: single crop ──────────────────────────────────
            float s_x = sz * ((float)cfg.instance_size / cfg.exemplar_size);
            scale_z_use = scale_z;
            win_inf = cfg.window_influence;

            double tp = now_ms();
            auto crop = get_subwindow(frame, cx, cy,
                                      cfg.instance_size, std::round(s_x), avg_chans);
            auto srch_t = to_nchw(crop);
            t_pre += now_ms() - tp;

            tp = now_ms();
            auto [xf, _bb] = srch_enc.run(srch_t, 3, cfg.instance_size, cfg.instance_size);
            t_backbone += now_ms() - tp;

            tp = now_ms();
            auto [scores, bboxes] = run_xcorr_head(
                xf, zf, anchors, AN, ss,
                0.0f, 0.0f, (float)cfg.instance_size, (float)cfg.instance_size,
                head_sess, head_in_c, head_out_c, mem);
            t_xcorr += now_ms() - tp;

            // Undo offset correction for normal mode (tile == full crop)
            // the correction adds 0 + inst/2 - inst/2 = 0, so no-op. ✓
            all_scores = scores;
            all_bboxes = bboxes;

            if (cfg.use_pf) {
                auto grid = PF::build_grid(scores, AN, ss);
                pf.predict(cfg.pf_std, pf_rng);
                pf.update(grid, (float)cx, (float)cy, scale_z, cfg.anchor_stride, ss);
                pf.resample(pf_rng);
            }

        } else {
            // ── LT mode: 2×2 tiled search ─────────────────────────────────
            float s_x = sz * ((float)cfg.instance_size / cfg.exemplar_size);
            s_x = std::max(s_x, std::max((float)cfg.effective_lost_size(),
                           (float)(cfg.instance_size * cfg.num_tiles) * 0.75f));
            int full_sz = (int)std::round(s_x);
            scale_z_use = 1.0f;
            win_inf = 0.001f;

            double tp = now_ms();
            cv::Mat crop_full = get_subwindow(frame, cx, cy, full_sz, full_sz, avg_chans);
            t_pre += now_ms() - tp;

            int tile_stride = (cfg.num_tiles > 1)
                ? (full_sz - cfg.instance_size) / (cfg.num_tiles - 1) : 0;

            // Pre-extract all tile NCHW buffers sequentially (CPU, before launch)
            struct TileJob { int tx, ty; std::vector<float> nchw; };
            std::vector<TileJob> jobs;
            jobs.reserve(N_TILE_WORKERS);
            for (int ti = 0; ti < cfg.num_tiles; ti++) {
                for (int tj = 0; tj < cfg.num_tiles; tj++) {
                    int tx = std::min(tj * tile_stride, full_sz - cfg.instance_size);
                    int ty = std::min(ti * tile_stride, full_sz - cfg.instance_size);

                    cv::Mat tile_mat;
                    cv::Rect roi(tx, ty, cfg.instance_size, cfg.instance_size);
                    if (roi.x + roi.width  <= crop_full.cols &&
                        roi.y + roi.height <= crop_full.rows) {
                        tile_mat = crop_full(roi).clone();
                    } else {
                        tile_mat = cv::Mat(cfg.instance_size, cfg.instance_size,
                                           CV_8UC3, cv::mean(crop_full));
                        cv::Rect valid_roi(0, 0,
                            std::min(roi.width,  crop_full.cols - tx),
                            std::min(roi.height, crop_full.rows - ty));
                        if (valid_roi.width > 0 && valid_roi.height > 0)
                            crop_full(cv::Rect(tx, ty, valid_roi.width, valid_roi.height))
                                .copyTo(tile_mat(valid_roi));
                    }
                    jobs.push_back({tx, ty, to_nchw(tile_mat)});
                }
            }

            // Distribute 4 tiles across 3 workers (round-robin: t%3).
            // Worker 0 runs tiles {0,3} sequentially; workers 1 and 2 each run
            // 1 tile, fully overlapping worker 0's first tile on separate cores.
            using WorkerResult = std::pair<std::vector<float>, std::vector<BBox>>;
            std::vector<std::vector<int>> worker_tiles(N_TILE_WORKERS);
            for (int t = 0; t < (int)jobs.size(); t++)
                worker_tiles[t % N_TILE_WORKERS].push_back(t);

            std::vector<std::future<WorkerResult>> futures;
            futures.reserve(N_TILE_WORKERS);

            tp = now_ms();
            for (int w = 0; w < N_TILE_WORKERS; w++) {
                futures.push_back(std::async(std::launch::async,
                    [&, w]() -> WorkerResult {
                        WorkerResult combined;
                        for (int t : worker_tiles[w]) {
                            auto [xf_tile, enc_ms] = tile_encs[w].run(
                                jobs[t].nchw, 3, cfg.instance_size, cfg.instance_size);
                            (void)enc_ms;
                            auto [ts, tb] = run_xcorr_head(
                                xf_tile, zf, anchors, AN, ss,
                                (float)jobs[t].tx, (float)jobs[t].ty,
                                (float)cfg.instance_size, (float)full_sz,
                                tile_head_sess[w], head_in_c, head_out_c, mem);
                            combined.first.insert(combined.first.end(), ts.begin(), ts.end());
                            combined.second.insert(combined.second.end(), tb.begin(), tb.end());
                        }
                        return combined;
                    }));
            }
            for (auto& f : futures) {
                auto [ts, tb] = f.get();
                all_scores.insert(all_scores.end(), ts.begin(), ts.end());
                all_bboxes.insert(all_bboxes.end(), tb.begin(), tb.end());
            }
            t_backbone += now_ms() - tp;  // wall-clock of parallel block
        }

        // ── Hann window for normal mode, flat 0.5 for LT ───────────────────
        std::vector<float> window;
        if (!longterm) {
            window = cosine_win;
        } else {
            window.assign(all_scores.size(), 0.5f);
        }

        auto best = select_best(all_scores, all_bboxes, window, win_inf,
                                 scale_z_use, tw, th, cfg.penalty_k, cfg.lr);

        // ── State update (Python update block) ─────────────────────────────
        float best_score = best.score;
        if (best_score >= cfg.conf_low) {
            if (!longterm && cfg.use_pf) {
                auto [pcx, pcy] = pf.estimate();
                cx = pcx;
                cy = pcy;
            } else {
                cx = best.bbox.cx + cx;
                cy = best.bbox.cy + cy;
            }
            tw = tw * (1 - best.lr) + best.bbox.w * best.lr;
            th = th * (1 - best.lr) + best.bbox.h * best.lr;
        } else if (cfg.use_pf) {
            // Low confidence — re-centre particles on the argmax prediction so
            // they don't drift unconstrained while the target is temporarily lost.
            pf.recenter((float)(best.bbox.cx + cx), (float)(best.bbox.cy + cy));
        }
        // Clamp to frame
        cx = std::max(0.0, std::min((double)VW, cx));
        cy = std::max(0.0, std::min((double)VH, cy));
        tw = std::max(10.0f, std::min((float)VW, tw));
        th = std::max(10.0f, std::min((float)VH, th));

        // ── LT mode transition ─────────────────────────────────────────────
        if (cfg.enable_lt_mode) {
            bool was_lt = longterm;
            if (best_score < cfg.conf_low)       longterm = true;
            else if (best_score > cfg.conf_high) longterm = false;
            if (was_lt && !longterm) {
                lt_stuck_frames = 0;
                if (cfg.use_pf) pf.recenter((float)cx, (float)cy);
            }
        }

        // ── Global scan: if stuck in LT, search all 4 frame quadrants ──────
        if (longterm && cfg.lt_stuck_timeout > 0) {
            if (best_score < cfg.lt_stuck_threshold) lt_stuck_frames++;
            else                                     lt_stuck_frames = 0;

            if (lt_stuck_frames == cfg.lt_stuck_timeout) {
                lt_stuck_frames = 0;
                float s_x_gs = sz * ((float)cfg.instance_size / cfg.exemplar_size);
                // 3×2 grid within the central 70% of the frame ([15%,85%] each axis).
                float x0 = 0.15f * VW, x1 = 0.85f * VW;
                float y0 = 0.15f * VH, y1 = 0.85f * VH;
                std::vector<std::pair<double,double>> scan_centers;
                for (int iy = 0; iy < 2; iy++)
                    for (int ix = 0; ix < 3; ix++)
                        scan_centers.push_back({
                            x0 + (x1-x0) * (ix + 0.5) / 3.0,
                            y0 + (y1-y0) * (iy + 0.5) / 2.0 });
                float best_gs = -1.f;
                double best_gcx = cx, best_gcy = cy;
                for (auto [gcx, gcy] : scan_centers) {
                    auto gc = get_subwindow(frame, gcx, gcy,
                                           cfg.instance_size, std::round(s_x_gs), avg_chans);
                    auto [xf_gs, _] = srch_enc.run(to_nchw(gc), 3,
                                                   cfg.instance_size, cfg.instance_size);
                    auto [sc_gs, _2] = run_xcorr_head(
                        xf_gs, zf, anchors, AN, ss,
                        0.f, 0.f, (float)cfg.instance_size, (float)cfg.instance_size,
                        head_sess, head_in_c, head_out_c, mem);
                    float qs = *std::max_element(sc_gs.begin(), sc_gs.end());
                    if (qs > best_gs) { best_gs = qs; best_gcx = gcx; best_gcy = gcy; }
                }
                fprintf(stderr, "f=%5d  Global scan → quadrant (%.0f,%.0f) score=%.3f\n",
                        fi, best_gcx, best_gcy, best_gs);
                if (best_gs > cfg.lt_stuck_threshold) {
                    cx = best_gcx; cy = best_gcy;
                    if (cfg.use_pf) pf.recenter((float)cx, (float)cy);
                }
            }
        }

        // ── Visualization ──────────────────────────────────────────────────
        int bx = (int)(cx - tw/2), by = (int)(cy - th/2);
        int bw = (int)tw,          bh = (int)th;
        const char* status = longterm ? "SEARCHING"
                           : (best_score > cfg.conf_low ? "TRACKING" : "LOW_CONF");
        cv::Scalar col = longterm ? cv::Scalar(0,165,255)
                       : (best_score > cfg.conf_low ? cv::Scalar(0,255,255)
                                                    : cv::Scalar(0,0,255));
        cv::rectangle(frame, {bx, by, bw, bh}, col, 2);
        cv::circle(frame, {(int)cx, (int)cy}, 4, col, -1);

        char label[64];
        snprintf(label, sizeof(label), "%s (%.2f)", status, best_score);
        int baseline;
        auto ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, {bx, by - ts.height - 10}, {bx + ts.width, by}, col, -1);
        cv::putText(frame, label, {bx, by - 5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1);

        char info[128];
        snprintf(info, sizeof(info), "f=%d  score=%.3f  LT=%s", fi, best_score, longterm?"ON":"OFF");
        cv::putText(frame, info, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);

        if (fi % 100 == 0)
            fprintf(stderr, "f=%5d  score=%.3f  cx=%.1f cy=%.1f w=%.1f h=%.1f  LT=%s\n",
                    fi, best_score, cx, cy, tw, th, longterm ? "ON" : "OFF");

        if (writer.isOpened()) writer.write(frame);
        if (cfg.display) {
            cv::imshow("poc_siamrpn++ [Q/ESC=quit]", frame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q' || key == 27) break;
        }
        fi++;
    }

    double elapsed_s = (now_ms() - t_wall_start) / 1000.0;
    int n = fi - 1;
    fprintf(stderr, "\nProcessed %d frames in %.1fs  →  %.1f fps\n", n, elapsed_s, n / elapsed_s);
    fprintf(stderr, "  preprocess:        %5.1f ms/frame\n", t_pre      / n);
    fprintf(stderr, "  search (Metis):    %5.1f ms/frame\n", t_backbone / n);
    fprintf(stderr, "  dw-xcorr+head:     %5.1f ms/frame\n", t_xcorr    / n);
    fprintf(stderr, "    ├ dw-xcorr only: %5.1f ms/frame  (main thread)\n", tl_t_xcorr_only / n);
    fprintf(stderr, "    └ head only:     %5.1f ms/frame  (main thread)\n", tl_t_head_only  / n);

    if (cfg.display) cv::destroyAllWindows();
    return 0;
}
