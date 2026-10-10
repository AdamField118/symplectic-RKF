/*
 * Runs geodesic.comp, the GLSL port of Bernard Ymeri's symplectic RK4/RK6
 * integrator (WPI MQP 2026, Appendix A), on the GPU.
 *
 * There's no window or X server involved. The OpenGL context comes from EGL,
 * which works on a headless GPU node. libEGL is opened at run time and every
 * GL function is looked up through it, so building only needs a C compiler:
 *
 *     cc -O2 -std=c11 -o geodesic main.c -ldl -lm
 *
 * The result is a NumPy .npy file of shape (n_geodesics, steps + 1, 8). Each
 * row is (t, r, theta, phi, p_t, p_r, p_theta, p_phi), so for one initial
 * condition you get the same array integrate_geodesic() returns in Python.
 *
 * Usage: ./geodesic [options]
 *   --M <float>          black hole mass                   (default 1.0)
 *   --a <float>          spin a = J/M; 0 gives Schwarzschild (default 0.5)
 *   --h <float>          step size                         (default 1)
 *   --steps <int>        number of steps                   (default 15000)
 *   --ic "t r th ph pt pr pth pph"   one initial condition; repeatable
 *                        (default: Appendix A's Kerr initial condition)
 *   --ic-file <path>     initial conditions, 8 numbers per line
 *   --out <path>         output .npy                       (default geodesic.npy)
 *   --shader <path>      compute shader (default: geodesic.comp next to the binary)
 *   --chunk <int>        steps per dispatch                (default 100)
 *   --selftest           check the shader's functions against a CPU version
 *                        of the same code, then exit (uses --M/--a/--h/--ic)
 *
 * Environment variables:
 *   EGL_LIBRARY=<path>   use this libEGL.so.1 instead of searching for one
 *   EGL_DEVICE=<n>       use the n-th GPU when more than one is visible
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* The few EGL and GL declarations we need, so no system headers are required. */
typedef void *EGLDisplay, *EGLConfig, *EGLContext, *EGLSurface, *EGLDeviceEXT;
typedef int32_t EGLint;
typedef unsigned int EGLBoolean, EGLenum;
#define EGL_NONE 0x3038
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_OPENGL_BIT 0x0008
#define EGL_OPENGL_API 0x30A2
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x00000001
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#define EGL_EXTENSIONS 0x3055
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_CONFIG_KHR ((EGLConfig)0)

typedef unsigned int GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef intptr_t GLintptr;
typedef ptrdiff_t GLsizeiptr;
#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_DYNAMIC_COPY 0x88EA
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02

static void *(*eglGetProcAddress_)(const char *);
#define EGLFN(ret, name, ...) static ret (*name##_)(__VA_ARGS__);
EGLFN(EGLDisplay, eglGetDisplay, void *)
EGLFN(EGLBoolean, eglInitialize, EGLDisplay, EGLint *, EGLint *)
EGLFN(EGLBoolean, eglBindAPI, EGLenum)
EGLFN(EGLBoolean, eglChooseConfig, EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *)
EGLFN(EGLContext, eglCreateContext, EGLDisplay, EGLConfig, EGLContext, const EGLint *)
EGLFN(EGLBoolean, eglMakeCurrent, EGLDisplay, EGLSurface, EGLSurface, EGLContext)
EGLFN(EGLint, eglGetError, void)
EGLFN(const char *, eglQueryString, EGLDisplay, EGLint)
EGLFN(EGLBoolean, eglTerminate, EGLDisplay)
EGLFN(EGLBoolean, eglQueryDevicesEXT, EGLint, EGLDeviceEXT *, EGLint *)
EGLFN(EGLDisplay, eglGetPlatformDisplayEXT, EGLenum, void *, const EGLint *)

#define GLFN(ret, name, ...) static ret (*name)(__VA_ARGS__);
GLFN(const GLubyte *, glGetString, GLenum)
GLFN(GLenum, glGetError, void)
GLFN(GLuint, glCreateShader, GLenum)
GLFN(void, glShaderSource, GLuint, GLsizei, const GLchar *const *, const GLint *)
GLFN(void, glCompileShader, GLuint)
GLFN(void, glGetShaderiv, GLuint, GLenum, GLint *)
GLFN(void, glGetShaderInfoLog, GLuint, GLsizei, GLsizei *, GLchar *)
GLFN(GLuint, glCreateProgram, void)
GLFN(void, glAttachShader, GLuint, GLuint)
GLFN(void, glLinkProgram, GLuint)
GLFN(void, glGetProgramiv, GLuint, GLenum, GLint *)
GLFN(void, glGetProgramInfoLog, GLuint, GLsizei, GLsizei *, GLchar *)
GLFN(void, glUseProgram, GLuint)
GLFN(GLint, glGetUniformLocation, GLuint, const GLchar *)
GLFN(void, glUniform1i, GLint, GLint)
GLFN(void, glGenBuffers, GLsizei, GLuint *)
GLFN(void, glBindBuffer, GLenum, GLuint)
GLFN(void, glBufferData, GLenum, GLsizeiptr, const void *, GLenum)
GLFN(void, glBindBufferBase, GLenum, GLuint, GLuint)
GLFN(void, glGetBufferSubData, GLenum, GLintptr, GLsizeiptr, void *)
GLFN(void, glDispatchCompute, GLuint, GLuint, GLuint)
GLFN(void, glMemoryBarrier, GLbitfield)
GLFN(void, glFinish, void)

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void *egl_sym(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p && eglGetProcAddress_) p = eglGetProcAddress_(name);
    return p;
}

static void *gl_sym(const char *name) {
    void *p = eglGetProcAddress_(name);
    if (!p) {
        fprintf(stderr, "error: missing GL entry point %s\n", name);
        exit(1);
    }
    return p;
}

/* Headless OpenGL context */
static void create_context(void) {
    /* Where to look, in order: EGL_LIBRARY if it's set, then the active conda
       environment (environment.yaml includes libEGL.so.1, which Turing's GPU
       nodes don't have), then the system. */
    char conda_egl[4096] = "";
    const char *conda = getenv("CONDA_PREFIX");
    if (conda && *conda) snprintf(conda_egl, sizeof conda_egl, "%s/lib/libEGL.so.1", conda);
    const char *names[] = {getenv("EGL_LIBRARY"), conda_egl, "libEGL.so.1", "libEGL.so"};
    void *lib = NULL;
    for (int i = 0; i < 4 && !lib; i++) {
        if (!names[i] || !*names[i]) continue;
        lib = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
        if (!lib) fprintf(stderr, "dlopen %s: %s\n", names[i], dlerror());
    }
    if (!lib)
        die("couldn't find libEGL.so.1. Activate the conda environment from "
            "environment.yaml first (see README).");

    eglGetProcAddress_ = (void *(*)(const char *))dlsym(lib, "eglGetProcAddress");
    if (!eglGetProcAddress_) die("libEGL has no eglGetProcAddress");
#define LOAD_EGL(name) name##_ = egl_sym(lib, #name)
    LOAD_EGL(eglGetDisplay); LOAD_EGL(eglInitialize); LOAD_EGL(eglBindAPI);
    LOAD_EGL(eglChooseConfig); LOAD_EGL(eglCreateContext); LOAD_EGL(eglMakeCurrent);
    LOAD_EGL(eglGetError); LOAD_EGL(eglQueryString); LOAD_EGL(eglTerminate);
    LOAD_EGL(eglQueryDevicesEXT); LOAD_EGL(eglGetPlatformDisplayEXT);

    EGLDisplay dpy = 0;
    EGLint major = 0, minor = 0;
    const char *how = "";

    /* First choice: ask EGL for a GPU directly (this is the path NVIDIA uses). */
    if (eglQueryDevicesEXT_ && eglGetPlatformDisplayEXT_) {
        EGLDeviceEXT devs[16];
        EGLint n = 0;
        if (eglQueryDevicesEXT_(16, devs, &n) && n > 0) {
            const char *env = getenv("EGL_DEVICE");
            int want = env ? atoi(env) : 0;
            for (int k = 0; k < n && !dpy; k++) {
                int i = (want + k) % n;
                EGLDisplay d = eglGetPlatformDisplayEXT_(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL);
                if (d && eglInitialize_(d, &major, &minor)) {
                    eglBindAPI_(EGL_OPENGL_API);
                    const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4,
                                               EGL_CONTEXT_MINOR_VERSION, 3,
                                               EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                               EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
                    EGLContext c = eglCreateContext_(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
                    if (c && eglMakeCurrent_(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) {
                        dpy = d;
                        how = "EGL device";
                    } else {
                        eglTerminate_(d);
                    }
                }
            }
        }
    }
    /* Fallbacks: Mesa's surfaceless platform, then the default display. */
    for (int attempt = 0; attempt < 2 && !dpy; attempt++) {
        EGLDisplay d = attempt == 0
            ? (eglGetPlatformDisplayEXT_ ? eglGetPlatformDisplayEXT_(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL) : 0)
            : eglGetDisplay_(NULL);
        if (!d || !eglInitialize_(d, &major, &minor)) continue;
        eglBindAPI_(EGL_OPENGL_API);
        const EGLint cfg_attr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
        EGLConfig cfg = EGL_NO_CONFIG_KHR;
        EGLint ncfg = 0;
        eglChooseConfig_(d, cfg_attr, &cfg, 1, &ncfg);
        const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3,
                                   EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                   EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
        EGLContext c = eglCreateContext_(d, ncfg ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
        if (c && eglMakeCurrent_(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) {
            dpy = d;
            how = attempt == 0 ? "EGL surfaceless" : "EGL default display";
        } else {
            eglTerminate_(d);
        }
    }
    if (!dpy) {
        fprintf(stderr, "EGL error 0x%x\n", eglGetError_ ? eglGetError_() : 0);
        die("could not create a headless OpenGL 4.3 core context");
    }

#define LOAD_GL(name) name = gl_sym(#name)
    LOAD_GL(glGetString); LOAD_GL(glGetError); LOAD_GL(glCreateShader);
    LOAD_GL(glShaderSource); LOAD_GL(glCompileShader); LOAD_GL(glGetShaderiv);
    LOAD_GL(glGetShaderInfoLog); LOAD_GL(glCreateProgram); LOAD_GL(glAttachShader);
    LOAD_GL(glLinkProgram); LOAD_GL(glGetProgramiv); LOAD_GL(glGetProgramInfoLog);
    LOAD_GL(glUseProgram); LOAD_GL(glGetUniformLocation); LOAD_GL(glUniform1i);
    LOAD_GL(glGenBuffers); LOAD_GL(glBindBuffer); LOAD_GL(glBufferData);
    LOAD_GL(glBindBufferBase); LOAD_GL(glGetBufferSubData); LOAD_GL(glDispatchCompute);
    LOAD_GL(glMemoryBarrier); LOAD_GL(glFinish);

    fprintf(stderr, "EGL %d.%d via %s\nGL_VENDOR   %s\nGL_RENDERER %s\nGL_VERSION  %s\n",
            major, minor, how, glGetString(GL_VENDOR), glGetString(GL_RENDERER),
            glGetString(GL_VERSION));
}

/* Shader */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) die("short read");
    s[n] = 0;
    fclose(f);
    return s;
}

static GLuint build_program_from_source(const char *src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, (const GLchar *const *)&src, NULL);
    glCompileShader(sh);
    GLint ok = 0, len = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
    if (len > 1) {
        char *log = malloc((size_t)len);
        glGetShaderInfoLog(sh, len, NULL, log);
        fprintf(stderr, "shader log:\n%s\n", log);
        free(log);
    }
    if (!ok) die("compute shader failed to compile");
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(prog, sizeof log, NULL, log);
        fprintf(stderr, "link log:\n%s\n", log);
        die("compute shader failed to link");
    }
    return prog;
}

static char *read_shader(const char *path) {
    char *src = read_file(path);
    if (!src) {
        fprintf(stderr, "error: cannot read shader %s\n", path);
        exit(1);
    }
    return src;
}

static GLuint build_program(const char *path) {
    char *src = read_shader(path);
    GLuint prog = build_program_from_source(src);
    free(src);
    return prog;
}

/* Writing the .npy file */
static long write_npy_header(FILE *f, long g, long rows) {
    char dict[256];
    int n = snprintf(dict, sizeof dict,
                     "{'descr': '<f8', 'fortran_order': False, 'shape': (%ld, %ld, 8), }", g, rows);
    int total = 10 + n + 1; /* 10 bytes of preamble, the dict, then a newline */
    int pad = (64 - total % 64) % 64;
    uint16_t hlen = (uint16_t)(n + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fputc(hlen & 0xff, f);
    fputc(hlen >> 8, f);
    fwrite(dict, 1, (size_t)n, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    return 10 + hlen;
}

/* main */
typedef struct {
    double M, a, h;
    double r4a11, r4a12, r4a21, r4a22, r4b1, r4b2;
    double r6a11, r6a12, r6a13, r6a21, r6a22, r6a23, r6a31, r6a32, r6a33;
    double r6b1, r6b2, r6b3;
} Params;

/* Self-test (--selftest)
 *
 * Runs the shader's own functions on the GPU and checks them against the same
 * code written in plain C on the CPU: sin/cos, division, sqrt, the metric, its
 * derivatives, the Hamiltonian, one RK4, RK6 and combined step, and 100 steps
 * in a row. Each line says how far apart the two are; anything flagged is
 * the part that's going wrong on this GPU. */

static double cM, ca;  /* black hole parameters for the C versions */

/* inverse_metric, dgdx, hamiltonian and the RK steps from Appendix A, in C */
static void c_inverse_metric(const double x[4], double g[4][4]) {
    double r = x[1], theta = x[2];
    double Sigma = r * r + (ca * ca) * (cos(theta) * cos(theta));
    double Delta = r * r - 2 * cM * r + ca * ca;
    double s2 = sin(theta) * sin(theta);
    memset(g, 0, sizeof(double) * 16);
    g[0][0] = -((r * r + ca * ca) * (r * r + ca * ca) - (ca * ca) * Delta * s2) / (Delta * Sigma);
    g[0][3] = g[3][0] = -2 * cM * ca * r / (Delta * Sigma);
    g[1][1] = Delta / Sigma;
    g[2][2] = 1.0 / Sigma;
    g[3][3] = (Delta - (ca * ca) * s2) / (Delta * Sigma * s2);
}

static void c_dgdx(const double x[4], double dg[4][4][4]) {
    const double delta = 1e-6;
    for (int w = 0; w < 4; w++) {
        double xp[4], xm[4], gp[4][4], gm[4][4];
        memcpy(xp, x, sizeof xp);
        memcpy(xm, x, sizeof xm);
        xp[w] = x[w] + delta;
        xm[w] = x[w] - delta;
        c_inverse_metric(xp, gp);
        c_inverse_metric(xm, gm);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) dg[w][i][j] = (gp[i][j] - gm[i][j]) / (2 * delta);
    }
}

static void c_hamiltonian(const double y[8], double d[8]) {
    double g[4][4], dg[4][4][4];
    c_inverse_metric(y, g);
    c_dgdx(y, dg);
    const double *p = y + 4;
    for (int i = 0; i < 4; i++) {
        double s = 0;
        for (int j = 0; j < 4; j++) s += g[i][j] * p[j];
        d[i] = s;
    }
    for (int w = 0; w < 4; w++) {
        double dp = 0;
        for (int u = 0; u < 4; u++)
            for (int v = 0; v < 4; v++) dp = dp - 0.5 * dg[w][u][v] * p[u] * p[v];
        d[4 + w] = dp;
    }
}

static double c_norm8(const double v[8]) {
    double s = 0;
    for (int i = 0; i < 8; i++) s += v[i] * v[i];
    return sqrt(s);
}

/* One implicit RK step with s stages (2 = RK4, 3 = RK6), as in Appendix A */
static void c_rk(int s, const double A[3][3], const double B[3], const double y[8], double h,
                 double out[8]) {
    double k[3][8], kn[3][8], yi[8], d[8];
    c_hamiltonian(y, k[0]);
    for (int i = 1; i < s; i++) memcpy(k[i], k[0], sizeof k[0]);
    for (int it = 0; it < 20; it++) {
        for (int i = 0; i < s; i++) {
            for (int c = 0; c < 8; c++) {
                double acc = A[i][0] * k[0][c];
                for (int j = 1; j < s; j++) acc = acc + A[i][j] * k[j][c];
                yi[c] = y[c] + h * acc;
            }
            c_hamiltonian(yi, kn[i]);
        }
        int done = 1;
        for (int i = 0; i < s; i++) {
            for (int c = 0; c < 8; c++) d[c] = kn[i][c] - k[i][c];
            if (!(c_norm8(d) < 5e-11)) done = 0;
        }
        if (done) break;
        memcpy(k, kn, sizeof k);
    }
    for (int c = 0; c < 8; c++) {
        double acc = B[0] * k[0][c];
        for (int j = 1; j < s; j++) acc = acc + B[j] * k[j][c];
        out[c] = y[c] + h * acc;
    }
}

static void c_rk46(const Params *P, const double y[8], double h, double out[8]) {
    const double A4[3][3] = {{P->r4a11, P->r4a12}, {P->r4a21, P->r4a22}}, B4[3] = {P->r4b1, P->r4b2};
    const double A6[3][3] = {{P->r6a11, P->r6a12, P->r6a13},
                             {P->r6a21, P->r6a22, P->r6a23},
                             {P->r6a31, P->r6a32, P->r6a33}};
    const double B6[3] = {P->r6b1, P->r6b2, P->r6b3};
    double hc = h, y6[8], d[8];
    for (int it = 0; it < 2; it++) {
        c_rk(2, A4, B4, y, hc, out);
        c_rk(3, A6, B6, y, hc, y6);
        for (int c = 0; c < 8; c++) d[c] = y6[c] - out[c];
        double error = c_norm8(d);
        if (error < 0.1) return;
        hc = 0.9 * hc * pow(0.1 / error, 1.0 / 5);
    }
}

/* Test entry point appended to geodesic.comp in place of its main() */
static const char *selftest_main =
    "\n"
    "layout(std430, binding = 3) buffer Test { double t[]; };\n"
    "// (Only constant indices into vectors: NVIDIA can't link v[i] = ... for doubles.)\n"
    "// Driver checks: hand doubles back three different ways\n"
    "void probe_out_array(out double v[8]) { for (int i = 0; i < 8; i++) v[i] = t[200 + i]; }\n"
    "void probe_out_dvec4(out dvec4 v) { v = dvec4(t[200], t[201], t[202], t[203]); }\n"
    "dmat4 probe_return_dmat4() {\n"
    "    dvec4 a = dvec4(t[200], t[201], t[202], t[203]), b = dvec4(t[204], t[205], t[206], t[207]);\n"
    "    return dmat4(a, b, a, b);\n"
    "}\n"
    "void put4(inout int o, dvec4 v) { t[o] = v[0]; t[o + 1] = v[1]; t[o + 2] = v[2]; t[o + 3] = v[3]; o += 4; }\n"
    "void put_q_p(inout int o, dvec4 q, dvec4 p) { put4(o, q); put4(o, p); }\n"
    "void put_mat(inout int o, dmat4 m) { put4(o, m[0]); put4(o, m[1]); put4(o, m[2]); put4(o, m[3]); }\n"
    "void put_metric_and_derivatives(inout int o, dvec4 x) {\n"
    "    put_mat(o, inverse_metric(x));\n"
    "    put_mat(o, dgdx(x, 0)); put_mat(o, dgdx(x, 1)); put_mat(o, dgdx(x, 2)); put_mat(o, dgdx(x, 3));\n"
    "}\n"
    "void main() {\n"
    "    if (gl_GlobalInvocationID.x != 0u) return;\n"
    "    int o = 256;  // inputs live in t[0..255], results from t[256] on\n"
    "    for (int i = 0; i < 32; i++) { t[o++] = dsin(t[i]); t[o++] = dcos(t[i]); }\n"
    "    for (int i = 0; i < 32; i++) { t[o++] = ddiv(t[32 + 2*i], t[33 + 2*i]); t[o++] = dsqrt(t[33 + 2*i]); }\n"
    "    dvec4 q = dvec4(state[0], state[1], state[2], state[3]);\n"
    "    dvec4 p = dvec4(state[4], state[5], state[6], state[7]);\n"
    "    dvec4 dq, dp, nq, np;\n"
    "    put_metric_and_derivatives(o, q);\n"
    "    hamiltonian(q, p, dq, dp);\n"
    "    put_q_p(o, dq, dp);\n"
    "    // a second, less special point (theta != pi/2)\n"
    "    dvec4 zq = dvec4(t[100], t[101], t[102], t[103]), zp = dvec4(t[104], t[105], t[106], t[107]);\n"
    "    put_metric_and_derivatives(o, zq);\n"
    "    hamiltonian(zq, zp, dq, dp);\n"
    "    put_q_p(o, dq, dp);\n"
    "    symplectic_rk4(q, p, h, 5e-11LF, 20, nq, np);\n"
    "    put_q_p(o, nq, np);\n"
    "    symplectic_rk6(q, p, h, 5e-11LF, 20, nq, np);\n"
    "    put_q_p(o, nq, np);\n"
    "    symplectic_rk46_comparison(q, p, h, nq, np);\n"
    "    put_q_p(o, nq, np);\n"
    "    for (int n = 0; n < 100; n++) { symplectic_rk46_comparison(q, p, h, nq, np); q = nq; p = np; }\n"
    "    put_q_p(o, q, p);\n"
    "    double v[8];\n"
    "    probe_out_array(v);\n"
    "    for (int i = 0; i < 8; i++) t[o++] = v[i];\n"
    "    dvec4 pv;\n"
    "    probe_out_dvec4(pv);\n"
    "    put4(o, pv);\n"
    "    put_mat(o, probe_return_dmat4());\n"
    "}\n";

/* Compares one group of GPU results with the CPU's and prints a line.
   Errors are relative to the largest value in the group. */
static int report(const char *name, const double *gpu, const double *cpu, int n, double limit) {
    double scale = 0, worst = 0;
    int at = 0;
    for (int i = 0; i < n; i++) scale = fmax(scale, fabs(cpu[i]));
    if (scale == 0) scale = 1;
    for (int i = 0; i < n; i++) {
        double e = fabs(gpu[i] - cpu[i]) / scale;
        if (!(e <= worst)) { worst = e; at = i; }
    }
    int ok = worst <= limit;
    printf("  %-28s %-4s  worst relative difference %.2e (allowed %.0e)", name, ok ? "ok" : "BAD", worst, limit);
    if (!ok) printf("  at #%d: gpu %.17g, cpu %.17g", at, gpu[at], cpu[at]);
    printf("\n");
    return ok;
}

static int selftest(const char *shader_path, const Params *P, const double y0[8]) {
    cM = P->M;
    ca = P->a;

    /* Swap the shader's main() for the test one */
    char *src = read_shader(shader_path);
    char *m = strstr(src, "void main()");
    if (!m) die("couldn't find main() in the shader");
    const char *renamed = "void integrate_main()";
    size_t len = strlen(src) + strlen(renamed) + strlen(selftest_main) + 1;
    char *test_src = malloc(len);
    snprintf(test_src, len, "%.*s%s%s%s", (int)(m - src), src, renamed,
             m + strlen("void main()"), selftest_main);
    GLuint prog = build_program_from_source(test_src);
    free(src);
    free(test_src);
    glUseProgram(prog);

    double in[256] = {0};
    for (int i = 0; i < 32; i++) in[i] = -3.0 + 0.2 * i + 0.0123;      /* angles for sin/cos */
    for (int i = 0; i < 32; i++) {
        in[32 + 2 * i] = 1.0 + 0.37 * i;                            /* a/b and sqrt(b) */
        in[33 + 2 * i] = 3.0 - 0.071 * i + 1e-3 * i * i;
    }
    const double z[8] = {0.0, 17.3, 1.1, 0.4, -0.97, 0.12, 3.5, 2.9};  /* the second point */
    memcpy(in + 100, z, sizeof z);
    for (int i = 0; i < 8; i++) in[200 + i] = (i + 1) / 3.0;  /* inputs for the driver checks */
    double *buf_data = calloc(1024, sizeof(double));
    memcpy(buf_data, in, sizeof in);

    GLuint buf[3];
    glGenBuffers(3, buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof *P, P, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf[0]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(double) * 8, y0, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf[1]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(double) * 1024, buf_data, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buf[2]);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(double) * 1024, buf_data);
    if (glGetError()) die("GL error during the self-test");
    const double *gpu = buf_data + 256;

    /* The same things on the CPU */
    double cpu[768] = {0};  /* results: 128 + 88 + 88 + 32 + 28 doubles */
    int o = 0;
    for (int i = 0; i < 32; i++) { cpu[o++] = sin(in[i]); cpu[o++] = cos(in[i]); }
    for (int i = 0; i < 32; i++) { cpu[o++] = in[32 + 2 * i] / in[33 + 2 * i]; cpu[o++] = sqrt(in[33 + 2 * i]); }
    double g[4][4], dg[4][4][4], d[8], s[8], y[8];
    c_inverse_metric(y0, g);
    memcpy(cpu + o, g, sizeof g); o += 16;
    c_dgdx(y0, dg);
    memcpy(cpu + o, dg, sizeof dg); o += 64;
    c_hamiltonian(y0, d);
    memcpy(cpu + o, d, sizeof d); o += 8;
    c_inverse_metric(z, g);
    memcpy(cpu + o, g, sizeof g); o += 16;
    c_dgdx(z, dg);
    memcpy(cpu + o, dg, sizeof dg); o += 64;
    c_hamiltonian(z, d);
    memcpy(cpu + o, d, sizeof d); o += 8;
    const double A4[3][3] = {{P->r4a11, P->r4a12}, {P->r4a21, P->r4a22}}, B4[3] = {P->r4b1, P->r4b2};
    const double A6[3][3] = {{P->r6a11, P->r6a12, P->r6a13},
                             {P->r6a21, P->r6a22, P->r6a23},
                             {P->r6a31, P->r6a32, P->r6a33}};
    const double B6[3] = {P->r6b1, P->r6b2, P->r6b3};
    c_rk(2, A4, B4, y0, P->h, s);
    memcpy(cpu + o, s, sizeof s); o += 8;
    c_rk(3, A6, B6, y0, P->h, s);
    memcpy(cpu + o, s, sizeof s); o += 8;
    c_rk46(P, y0, P->h, s);
    memcpy(cpu + o, s, sizeof s); o += 8;
    memcpy(y, y0, sizeof y);
    for (int n = 0; n < 100; n++) { c_rk46(P, y, P->h, s); memcpy(y, s, sizeof y); }
    memcpy(cpu + o, y, sizeof y); o += 8;
    for (int i = 0; i < 8; i++) cpu[o++] = in[200 + i];                        /* out double[8] */
    for (int i = 0; i < 4; i++) cpu[o++] = in[200 + i];                        /* out dvec4 */
    for (int i = 0; i < 16; i++) cpu[o++] = in[200 + i % 8];                   /* return dmat4 */

    /* sin, cos, division, sqrt and the metric should agree to the last digit
       or two. Derivatives magnify those last digits about a million times, and
       the RK steps add the fixed-point iteration's 5e-11 tolerance. */
    printf("self-test: GPU functions vs. the same code on the CPU\n");
    int ok = 1, k = 0;
    double gs[32], cs[32], gc[32], cc[32], gd[32], cd[32], gq[32], cq[32];
    for (int i = 0; i < 32; i++) {
        gs[i] = gpu[k + 2 * i]; cs[i] = cpu[k + 2 * i];
        gc[i] = gpu[k + 2 * i + 1];     cc[i] = cpu[k + 2 * i + 1];
    }
    k += 64;
    for (int i = 0; i < 32; i++) {
        gd[i] = gpu[k + 2 * i]; cd[i] = cpu[k + 2 * i];
        gq[i] = gpu[k + 2 * i + 1]; cq[i] = cpu[k + 2 * i + 1];
    }
    k += 64;
    ok &= report("sin", gs, cs, 32, 1e-15);
    ok &= report("cos", gc, cc, 32, 1e-15);
    ok &= report("division (ddiv)", gd, cd, 32, 1e-15);
    ok &= report("square root (dsqrt)", gq, cq, 32, 1e-15);
    ok &= report("inverse_metric", gpu + k, cpu + k, 16, 1e-14); k += 16;
    ok &= report("dgdx", gpu + k, cpu + k, 64, 1e-6); k += 64;
    ok &= report("hamiltonian", gpu + k, cpu + k, 8, 1e-7); k += 8;
    ok &= report("inverse_metric (2nd point)", gpu + k, cpu + k, 16, 1e-14); k += 16;
    ok &= report("dgdx (2nd point)", gpu + k, cpu + k, 64, 1e-6); k += 64;
    ok &= report("hamiltonian (2nd point)", gpu + k, cpu + k, 8, 1e-7); k += 8;
    ok &= report("symplectic_rk4 (1 step)", gpu + k, cpu + k, 8, 1e-8); k += 8;
    ok &= report("symplectic_rk6 (1 step)", gpu + k, cpu + k, 8, 1e-8); k += 8;
    ok &= report("rk46_comparison (1 step)", gpu + k, cpu + k, 8, 1e-8); k += 8;
    ok &= report("rk46_comparison (100 steps)", gpu + k, cpu + k, 8, 1e-6); k += 8;
    printf("driver checks (how this driver hands back doubles; the integrator only\n"
           "uses dvec4 and dmat4, so a BAD on the first line alone is expected):\n");
    report("out double[8] parameter", gpu + k, cpu + k, 8, 1e-15); k += 8;
    ok &= report("out dvec4 parameter", gpu + k, cpu + k, 4, 1e-15); k += 4;
    ok &= report("returned dmat4", gpu + k, cpu + k, 16, 1e-15); k += 16;
    printf(ok ? "all checks passed\n" : "some checks FAILED (marked BAD above)\n");
    free(buf_data);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    double M = 1.0, a = 0.5, h = 1.0;
    long steps = 15000, chunk = 100;
    const char *out = "geodesic.npy", *shader = NULL, *ic_file = NULL;
    double *ics = NULL;
    long n_ic = 0;
    int run_selftest = 0;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(k, "-h") || !strcmp(k, "--help")) {
            fprintf(stderr, "see the comment at the top of main.c for options\n");
            return 0;
        }
        if (!strcmp(k, "--selftest")) {
            run_selftest = 1;
            continue;
        }
        if (!v) die("option without a value");
        i++;
        if (!strcmp(k, "--M")) M = atof(v);
        else if (!strcmp(k, "--a")) a = atof(v);
        else if (!strcmp(k, "--h")) h = atof(v);
        else if (!strcmp(k, "--steps")) steps = atol(v);
        else if (!strcmp(k, "--chunk")) chunk = atol(v);
        else if (!strcmp(k, "--out")) out = v;
        else if (!strcmp(k, "--shader")) shader = v;
        else if (!strcmp(k, "--ic-file")) ic_file = v;
        else if (!strcmp(k, "--ic")) {
            ics = realloc(ics, sizeof(double) * 8 * (size_t)(n_ic + 1));
            if (sscanf(v, "%lf %lf %lf %lf %lf %lf %lf %lf", &ics[8 * n_ic], &ics[8 * n_ic + 1],
                       &ics[8 * n_ic + 2], &ics[8 * n_ic + 3], &ics[8 * n_ic + 4],
                       &ics[8 * n_ic + 5], &ics[8 * n_ic + 6], &ics[8 * n_ic + 7]) != 8)
                die("--ic needs 8 numbers");
            n_ic++;
        } else {
            fprintf(stderr, "unknown option %s\n", k);
            return 1;
        }
    }
    if (ic_file) {
        FILE *f = fopen(ic_file, "r");
        if (!f) die("cannot open --ic-file");
        double v[8];
        while (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3], &v[4],
                      &v[5], &v[6], &v[7]) == 8) {
            ics = realloc(ics, sizeof(double) * 8 * (size_t)(n_ic + 1));
            memcpy(&ics[8 * n_ic++], v, sizeof v);
        }
        fclose(f);
    }
    if (n_ic == 0) {
        /* Appendix A's initial conditions */
        static double y0[8] = {0.0, 20.0, M_PI / 2, 0.0, -0.9764550153430405, 0.0, 3.8, 3.0};
        ics = malloc(sizeof y0);
        memcpy(ics, y0, sizeof y0);
        n_ic = 1;
    }
    if (chunk < 1) chunk = 1;
    if (chunk > steps && steps > 0) chunk = steps;

    char shader_path[4096];
    if (!shader) {
        const char *slash = strrchr(argv[0], '/');
        int dirlen = slash ? (int)(slash - argv[0] + 1) : 0;
        snprintf(shader_path, sizeof shader_path, "%.*sgeodesic.comp", dirlen, argv[0]);
        shader = shader_path;
    }

    /* RK coefficients, written the same way as in the Python module */
    Params P = {
        .M = M, .a = a, .h = h,
        .r4a11 = 1.0 / 4, .r4a12 = 1.0 / 4 - sqrt(3.0) / 6,
        .r4a21 = 1.0 / 4 + sqrt(3.0) / 6, .r4a22 = 1.0 / 4,
        .r4b1 = 1.0 / 2, .r4b2 = 1.0 / 2,
        .r6a11 = 5.0 / 36, .r6a12 = 2.0 / 9 - sqrt(15.0) / 15, .r6a13 = 5.0 / 36 - sqrt(15.0) / 30,
        .r6a21 = 5.0 / 36 + sqrt(15.0) / 24, .r6a22 = 2.0 / 9, .r6a23 = 5.0 / 36 - sqrt(15.0) / 24,
        .r6a31 = 5.0 / 36 + sqrt(15.0) / 30, .r6a32 = 2.0 / 9 + sqrt(15.0) / 15, .r6a33 = 5.0 / 36,
        .r6b1 = 5.0 / 18, .r6b2 = 4.0 / 9, .r6b3 = 5.0 / 18,
    };

    create_context();
    if (run_selftest) return selftest(shader, &P, ics);
    GLuint prog = build_program(shader);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "n_geodesics"), (GLint)n_ic);

    GLuint buf[3];
    glGenBuffers(3, buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof P, &P, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf[0]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(double) * 8 * n_ic), ics, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf[1]);
    size_t chunk_bytes = sizeof(double) * 8 * (size_t)n_ic * (size_t)chunk;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)chunk_bytes, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buf[2]);
    if (glGetError() != 0) die("buffer setup failed (too many geodesics x chunk?)");

    FILE *f = fopen(out, "wb");
    if (!f) die("cannot open output file");
    long rows = steps + 1;
    long hdr = write_npy_header(f, n_ic, rows);
    /* Row 0 is the starting point, like geodesic.append(y.copy()) in Python */
    for (long g = 0; g < n_ic; g++) {
        fseek(f, hdr + (long)sizeof(double) * 8 * (g * rows), SEEK_SET);
        fwrite(&ics[8 * g], sizeof(double), 8, f);
    }

    double *host = malloc(chunk_bytes);
    GLint loc_chunk = glGetUniformLocation(prog, "chunk");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long done = 0; done < steps;) {
        long c = steps - done < chunk ? steps - done : chunk;
        glUniform1i(loc_chunk, (GLint)c);
        glDispatchCompute((GLuint)((n_ic + 63) / 64), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[2]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(sizeof(double) * 8 * n_ic * c), host);
        GLenum err = glGetError();
        if (err) {
            fprintf(stderr, "GL error 0x%x after step %ld\n", err, done);
            return 1;
        }
        /* The shader writes [geodesic][step][8], c steps per geodesic */
        for (long g = 0; g < n_ic; g++) {
            fseek(f, hdr + (long)sizeof(double) * 8 * (g * rows + 1 + done), SEEK_SET);
            fwrite(&host[8 * g * c], sizeof(double), (size_t)(8 * c), f);
        }
        done += c;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
        fprintf(stderr, "\rstep %ld/%ld  %.1f s", done, steps, el);
    }
    glFinish();
    fprintf(stderr, "\nwrote %s  shape (%ld, %ld, 8)\n", out, n_ic, rows);
    fclose(f);
    free(host);
    free(ics);
    return 0;
}
