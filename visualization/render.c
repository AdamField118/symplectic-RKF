/*
 * Renders a video of a Kerr black hole with blackhole.comp, which traces every
 * pixel's light ray with Bernard Ymeri's symplectic RK4/RK6 integrator. The
 * picture and the camera's flight path are from blackhole-bg
 * (https://codeberg.org/AdamField118/blackhole-bg): an orbit that sets off up
 * and to the left out of the disk, goes all the way round, and comes back to
 * where it started.
 *
 * Like ../glsl/main.c it needs no window: the OpenGL context comes from EGL.
 * Frames are piped to ffmpeg, which writes the video. With --serve it draws
 * frames on request for the live viewer (viewer.py) instead; see serve().
 *
 *     cc -O2 -std=c11 -pthread -o blackhole render.c -ldl -lm
 *
 * Usage: ./blackhole [options]
 *   --out <path>        video to write; a .png writes one frame (default blackhole.mp4)
 *   --width <int>       (default 1280)
 *   --height <int>      (default 720)
 *   --fps <float>       (default 30)
 *   --seconds <float>   length of the flythrough (default 8)
 *   --frame <int>       with a .png --out, which frame to draw (default 0)
 *   --M <float>         black hole mass (default 1)
 *   --a <float>         spin; 0 is Schwarzschild (default 0.5, as in Appendix A)
 *   --distance <float>  camera distance from the hole, in units of M (default 20)
 *   --fov <float>       vertical field of view in degrees (default 60)
 *   --step <float>      each integration step starts at h = step * r (default 0.08)
 *   --tol <float>       RK4/RK6 comparison tolerance (default 1e-3)
 *   --rk-tol <float>    fixed-point iteration tolerance (default 1e-5)
 *   --rk-max-iter <int> fixed-point iteration limit (default 20, as in Appendix A)
 *   --max-steps <int>   steps before a ray gives up (default 2000)
 *   --disk-speed <float> units of M of time that pass per second of video (default 8)
 *   --no-disk           no accretion disk
 *   --no-doppler        disk colours without Doppler shift and beaming
 *   --sky <path>        sky image: raw 2048x2048 RGB8, as blackhole-bg bakes it
 *                       (default: a generated starfield)
 *   --shader <path>     compute shader (default: blackhole.comp next to the binary)
 *   --ffmpeg <path>     ffmpeg executable (default ffmpeg)
 *   --serve             draw frames on request for viewer.py (see serve())
 *
 * Environment variables:
 *   EGL_LIBRARY=<path>   use this libEGL.so.1 instead of searching for one
 *   EGL_DEVICE=<n>       use the n-th GPU when more than one is visible
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
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
typedef unsigned char GLboolean;
typedef float GLfloat;
typedef char GLchar;
typedef unsigned char GLubyte;
#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_TEXTURE_2D 0x0DE1
#define GL_TEXTURE0 0x84C0
#define GL_RGB 0x1907
#define GL_RGBA 0x1908
#define GL_RGB8 0x8051
#define GL_RGBA8 0x8058
#define GL_UNSIGNED_BYTE 0x1401
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_LINEAR 0x2601
#define GL_LINEAR_MIPMAP_LINEAR 0x2703
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_UNPACK_ALIGNMENT 0x0CF5
#define GL_PACK_ALIGNMENT 0x0D05
#define GL_WRITE_ONLY 0x88B9
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#define GL_TEXTURE_UPDATE_BARRIER_BIT 0x00000100

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
GLFN(void, glUniform1f, GLint, GLfloat)
GLFN(void, glUniform3f, GLint, GLfloat, GLfloat, GLfloat)
GLFN(void, glGenTextures, GLsizei, GLuint *)
GLFN(void, glDeleteTextures, GLsizei, const GLuint *)
GLFN(void, glBindTexture, GLenum, GLuint)
GLFN(void, glActiveTexture, GLenum)
GLFN(void, glTexStorage2D, GLenum, GLsizei, GLenum, GLsizei, GLsizei)
GLFN(void, glTexSubImage2D, GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void *)
GLFN(void, glTexParameteri, GLenum, GLenum, GLint)
GLFN(void, glGenerateMipmap, GLenum)
GLFN(void, glPixelStorei, GLenum, GLint)
GLFN(void, glBindImageTexture, GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum)
GLFN(void, glGetTexImage, GLenum, GLint, GLenum, GLenum, void *)
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
    LOAD_GL(glUniform1f); LOAD_GL(glUniform3f); LOAD_GL(glGenTextures);
    LOAD_GL(glDeleteTextures);
    LOAD_GL(glBindTexture); LOAD_GL(glActiveTexture); LOAD_GL(glTexStorage2D);
    LOAD_GL(glTexSubImage2D); LOAD_GL(glTexParameteri); LOAD_GL(glGenerateMipmap);
    LOAD_GL(glPixelStorei); LOAD_GL(glBindImageTexture); LOAD_GL(glGetTexImage);
    LOAD_GL(glDispatchCompute); LOAD_GL(glMemoryBarrier); LOAD_GL(glFinish);

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

static GLuint build_program(const char *path) {
    char *src = read_file(path);
    if (!src) {
        fprintf(stderr, "error: cannot read shader %s\n", path);
        exit(1);
    }
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
    free(src);
    return prog;
}


/* The sky */
#define SKY_SIZE 2048

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static double rnd(void) {
    /* splitmix64 */
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (double)((z ^ (z >> 31)) >> 11) / 9007199254740992.0;
}

/* A starfield to use when there's no --sky image: a dark background with a
   little colour in it, and stars of every brightness, mostly faint. It's the
   same every time. */
static unsigned char *make_sky(void) {
    const int n = SKY_SIZE;
    float *img = calloc((size_t)n * n * 3, sizeof(float));
    unsigned char *out = malloc((size_t)n * n * 3);
    if (!img || !out) die("out of memory");

    /* Faint clouds: a few smooth bumps of colour */
    for (int k = 0; k < 40; k++) {
        double cx = rnd() * n, cy = rnd() * n, rad = (0.05 + 0.15 * rnd()) * n;
        double amp = 0.006 + 0.012 * rnd();
        double col[3] = {0.5 + 0.5 * rnd(), 0.3 + 0.3 * rnd(), 0.6 + 0.4 * rnd()};
        int x0 = (int)fmax(0, cx - 3 * rad), x1 = (int)fmin(n - 1, cx + 3 * rad);
        int y0 = (int)fmax(0, cy - 3 * rad), y1 = (int)fmin(n - 1, cy + 3 * rad);
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                double d2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (rad * rad);
                double v = amp * exp(-d2);
                for (int c = 0; c < 3; c++) img[3 * ((size_t)y * n + x) + c] += (float)(v * col[c]);
            }
    }

    /* Stars. Brightness follows a power law, colour runs from blue-white to orange. */
    static const double tint[5][3] = {{0.75, 0.85, 1.0}, {0.9, 0.95, 1.0}, {1.0, 1.0, 1.0},
                                      {1.0, 0.92, 0.75}, {1.0, 0.75, 0.5}};
    for (int k = 0; k < 24000; k++) {
        double cx = rnd() * n, cy = rnd() * n;
        double flux = 0.04 * pow(1.0 - rnd(), -1.1);
        if (flux > 12.0) flux = 12.0;
        double sigma = 0.6 + 0.25 * log1p(flux);
        const double *t = tint[(int)(rnd() * 5)];
        int rad = (int)ceil(4 * sigma);
        for (int y = (int)cy - rad; y <= (int)cy + rad; y++)
            for (int x = (int)cx - rad; x <= (int)cx + rad; x++) {
                if (x < 0 || y < 0 || x >= n || y >= n) continue;
                double d2 = (x + 0.5 - cx) * (x + 0.5 - cx) + (y + 0.5 - cy) * (y + 0.5 - cy);
                double v = flux * exp(-d2 / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                for (int c = 0; c < 3; c++) img[3 * ((size_t)y * n + x) + c] += (float)(v * t[c]);
            }
    }

    for (size_t i = 0; i < (size_t)n * n; i++) {
        static const float base[3] = {5, 8, 13};
        for (int c = 0; c < 3; c++) {
            float v = base[c] + 255.0f * img[3 * i + c];
            out[3 * i + c] = (unsigned char)(v > 255 ? 255 : v);
        }
    }
    free(img);
    return out;
}

static void load_sky(const char *path) {
    const size_t want = (size_t)SKY_SIZE * SKY_SIZE * 3;
    unsigned char *pixels;
    if (path) {
        pixels = malloc(want);
        FILE *f = fopen(path, "rb");
        if (!pixels || !f || fread(pixels, 1, want, f) != want)
            die("--sky must be a raw 2048x2048 RGB8 file");
        fclose(f);
    } else {
        pixels = make_sky();
    }

    /* Mipmapped, because strongly lensed rays sample a blurrier level */
    GLuint tex;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    int levels = 1;
    for (int n = SKY_SIZE; n > 1; n >>= 1) levels++;
    glTexStorage2D(GL_TEXTURE_2D, levels, GL_RGB8, SKY_SIZE, SKY_SIZE);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SKY_SIZE, SKY_SIZE, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    free(pixels);
}

/* The camera's flight path, from blackhole-bg's animation_tick. Positions
   there have y as the disk's normal; here it's z, so (x, y, z) there is
   (x, -z, y) here. */
static void cross3(const double u[3], const double v[3], double out[3]) {
    out[0] = u[1] * v[2] - u[2] * v[1];
    out[1] = u[2] * v[0] - u[0] * v[2];
    out[2] = u[0] * v[1] - u[1] * v[0];
}

static void normalize3(double v[3]) {
    double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; i++) v[i] /= len;
}

static double ease_in_out(double t) {
    return t < 0.5 ? 2.0 * t * t : 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0;
}

/* t runs from 0 to 1 over the flythrough. Gives the camera's (r, theta, phi). */
static void flythrough(double t, double radius, double out[3]) {
    const double elevation = 0.525 * M_PI, azimuth = 0.0;
    double pos0[3] = {radius * sin(elevation) * cos(azimuth), radius * cos(elevation),
                      radius * sin(elevation) * sin(azimuth)};

    /* Camera basis looking at the origin with y up */
    double forward[3] = {-pos0[0] / radius, -pos0[1] / radius, -pos0[2] / radius};
    double up_world[3] = {0, 1, 0}, right[3], up[3];
    cross3(forward, up_world, right);
    normalize3(right);
    cross3(right, forward, up);

    /* Rotate about the axis that sends the camera 45 degrees up and to the left */
    double rhat[3] = {-forward[0], -forward[1], -forward[2]};
    double desired[3] = {-right[0] + up[0], -right[1] + up[1], -right[2] + up[2]};
    double axis[3];
    cross3(rhat, desired, axis);
    normalize3(axis);

    double angle = ease_in_out(t) * 2.0 * M_PI;
    double c = cos(angle), s = sin(angle);
    double dot = axis[0] * pos0[0] + axis[1] * pos0[1] + axis[2] * pos0[2];
    double cr[3], pos[3];
    cross3(axis, pos0, cr);
    for (int i = 0; i < 3; i++) pos[i] = pos0[i] * c + cr[i] * s + axis[i] * dot * (1.0 - c);

    double x = pos[0], y = -pos[2], z = pos[1];
    out[0] = sqrt(x * x + y * y + z * z);
    out[1] = acos(z / out[0]);
    out[2] = atan2(y, x);
}

/* Innermost stable circular orbit for prograde orbits (Bardeen, Press and Teukolsky 1972) */
static double isco(double M, double a) {
    double chi = a / M;
    double z1 = 1 + cbrt(1 - chi * chi) * (cbrt(1 + chi) + cbrt(1 - chi));
    double z2 = sqrt(3 * chi * chi + z1 * z1);
    return M * (3 + z2 - copysign(sqrt((3 - z1) * (3 + z1 + 2 * z2)), chi));
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int ends_with(const char *s, const char *suffix) {
    size_t n = strlen(s), k = strlen(suffix);
    return n >= k && !strcmp(s + n - k, suffix);
}

/* Frames go to ffmpeg from a second thread, so the GPU can draw the next
   frame while ffmpeg takes the last one. There are two frame buffers: the
   main thread fills one while the writer empties the other. */
static struct {
    pthread_mutex_t lock;
    pthread_cond_t changed;
    unsigned char *buf[2];
    int full[2];
    int frames;
    size_t bytes;
    FILE *pipe;
    int failed;
} W = {.lock = PTHREAD_MUTEX_INITIALIZER, .changed = PTHREAD_COND_INITIALIZER};

static void *writer(void *arg) {
    (void)arg;
    for (int f = 0; f < W.frames; f++) {
        int slot = f % 2;
        pthread_mutex_lock(&W.lock);
        while (!W.full[slot]) pthread_cond_wait(&W.changed, &W.lock);
        pthread_mutex_unlock(&W.lock);
        if (!W.failed && fwrite(W.buf[slot], 1, W.bytes, W.pipe) != W.bytes) W.failed = 1;
        pthread_mutex_lock(&W.lock);
        W.full[slot] = 0;
        pthread_cond_signal(&W.changed);
        pthread_mutex_unlock(&W.lock);
    }
    return NULL;
}

/* Drawing */
typedef struct {
    double M, a, fov, step, tol, rk_tol;
    int rk_max_iter, max_steps, disk, doppler;
} Settings;

static GLuint prog, target;
static int target_w, target_h;
static GLint loc_camera, loc_time, loc_row, loc_escape;

/* (Re)makes the texture the shader draws into */
static void make_target(int width, int height) {
    if (target && width == target_w && height == target_h) return;
    if (target) glDeleteTextures(1, &target);
    glGenTextures(1, &target);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, target);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
    glBindImageTexture(0, target, 0, 0, 0, GL_WRITE_ONLY, GL_RGBA8);
    target_w = width;
    target_h = height;
}

#define U(name) glGetUniformLocation(prog, name)

static void apply_settings(const Settings *S, int width, int height, int verbose) {
    double M = S->M, a = S->a;
    double r_plus = M + sqrt(M * M - a * a);
    double r_isco = isco(M, a);
    /* Innermost (prograde, equatorial) photon orbit */
    double r_photon = 2 * M * (1 + cos(2.0 / 3.0 * acos(-fabs(a) / M)));
    glUniform1i(U("starTex"), 1);
    glUniform1f(U("M"), (float)M);
    glUniform1f(U("a"), (float)a);
    glUniform1f(U("tanHalfFov"), (float)tan(S->fov * M_PI / 360.0));
    glUniform1f(U("aspect"), (float)width / height);
    glUniform1i(U("maxSteps"), S->max_steps);
    glUniform1f(U("stepScale"), (float)S->step);
    glUniform1f(U("tol"), (float)S->tol);
    glUniform1f(U("rkTol"), (float)S->rk_tol);
    glUniform1i(U("rkMaxIter"), S->rk_max_iter);
    glUniform1f(U("rHorizon"), (float)(1.01 * r_plus));
    glUniform1f(U("rCapture"), (float)r_photon);
    glUniform1f(U("diskInner"), (float)r_isco);
    glUniform1f(U("diskOuter"), (float)(13.0 * M));
    glUniform1i(U("diskEnabled"), S->disk);
    glUniform1i(U("doppler"), S->doppler);
    if (verbose)
        fprintf(stderr, "a = %g M: horizon at r = %.4f M, innermost photon orbit at %.4f M, "
                "disk from %.4f M to %g M\n", a / M, r_plus / M, r_photon / M, r_isco / M, 13.0);
}

/* Draws one frame into the target and waits for it. Returns the time it took. */
static double draw(const double cam[3], double time) {
    glUniform3f(loc_camera, (float)cam[0], (float)cam[1], (float)cam[2]);
    glUniform1f(loc_time, (float)time);
    /* Rays moving outward past 3 times the camera's distance have escaped */
    glUniform1f(loc_escape, (float)(3.0 * cam[0]));

    /* The frame is drawn in bands of rows, so that no single dispatch runs
       long enough to trip a desktop GPU's watchdog. */
    const int band = 128;
    double t0 = now();
    for (int row = 0; row < target_h; row += band) {
        glUniform1i(loc_row, row);
        int rows = target_h - row < band ? target_h - row : band;
        glDispatchCompute((GLuint)(target_w + 7) / 8, (GLuint)(rows + 7) / 8, 1);
    }
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
    glFinish();
    return now() - t0;
}

/* The live viewer's side of things. viewer.py sends one command per line on
   stdin and reads the answers on stdout:

     frame W H r theta phi time a fov step disk doppler
         Draws a W x H frame with the camera at (r, theta, phi) and the gas
         at coordinate time `time`. Answers with W and H (two int32s), the
         time drawing took in milliseconds (a float32), then the picture as
         W * H * 3 bytes of RGB, top row first.
     path t r
         Where the flythrough's camera is at t (0 to 1) when it flies at
         radius r. Answers with r, theta and phi as three float64s.

   It stops at the end of stdin. */
static int serve(Settings S) {
    char line[1024];
    unsigned char *pixels = NULL;
    size_t capacity = 0;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    while (fgets(line, sizeof line, stdin)) {
        int w, h, disk, doppler;
        double cam[3], time, a, fov, step, t, r;
        if (sscanf(line, "frame %d %d %lf %lf %lf %lf %lf %lf %lf %d %d", &w, &h, &cam[0],
                   &cam[1], &cam[2], &time, &a, &fov, &step, &disk, &doppler) == 11) {
            if (w < 8 || h < 8 || w > 8192 || h > 8192) die("bad frame size");
            S.a = fmax(-0.999 * S.M, fmin(0.999 * S.M, a));
            S.fov = fov;
            S.step = step;
            S.disk = disk;
            S.doppler = doppler;
            make_target(w, h);
            apply_settings(&S, w, h, 0);
            float ms = (float)(1e3 * draw(cam, time));
            size_t bytes = (size_t)w * h * 3;
            if (bytes > capacity) {
                pixels = realloc(pixels, bytes);
                if (!pixels) die("out of memory");
                capacity = bytes;
            }
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
            int32_t size[2] = {w, h};
            fwrite(size, sizeof size, 1, stdout);
            fwrite(&ms, sizeof ms, 1, stdout);
            fwrite(pixels, 1, bytes, stdout);
        } else if (sscanf(line, "path %lf %lf", &t, &r) == 2) {
            flythrough(t, r, cam);
            fwrite(cam, sizeof cam, 1, stdout);
        } else {
            fprintf(stderr, "unknown command: %s", line);
            return 1;
        }
        fflush(stdout);
    }
    return 0;
}

/* main */
int main(int argc, char **argv) {
    int width = 1280, height = 720, frame_only = 0, max_steps = 2000, rk_max_iter = 20;
    double fps = 30, seconds = 8, M = 1.0, a = 0.5, distance = 20, fov = 60;
    double step = 0.08, tol = 1e-3, rk_tol = 1e-5, disk_speed = 8;
    int disk = 1, doppler = 1, serving = 0;
    const char *out = "blackhole.mp4", *sky = NULL, *shader = NULL, *ffmpeg = "ffmpeg";

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        if (!strcmp(k, "-h") || !strcmp(k, "--help")) {
            fprintf(stderr, "see the comment at the top of render.c for options\n");
            return 0;
        }
        if (!strcmp(k, "--no-disk")) { disk = 0; continue; }
        if (!strcmp(k, "--no-doppler")) { doppler = 0; continue; }
        if (!strcmp(k, "--serve")) { serving = 1; continue; }
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) die("option without a value");
        i++;
        if (!strcmp(k, "--out")) out = v;
        else if (!strcmp(k, "--width")) width = atoi(v);
        else if (!strcmp(k, "--height")) height = atoi(v);
        else if (!strcmp(k, "--fps")) fps = atof(v);
        else if (!strcmp(k, "--seconds")) seconds = atof(v);
        else if (!strcmp(k, "--frame")) frame_only = atoi(v);
        else if (!strcmp(k, "--M")) M = atof(v);
        else if (!strcmp(k, "--a")) a = atof(v);
        else if (!strcmp(k, "--distance")) distance = atof(v);
        else if (!strcmp(k, "--fov")) fov = atof(v);
        else if (!strcmp(k, "--step")) step = atof(v);
        else if (!strcmp(k, "--tol")) tol = atof(v);
        else if (!strcmp(k, "--rk-tol")) rk_tol = atof(v);
        else if (!strcmp(k, "--rk-max-iter")) rk_max_iter = atoi(v);
        else if (!strcmp(k, "--max-steps")) max_steps = atoi(v);
        else if (!strcmp(k, "--disk-speed")) disk_speed = atof(v);
        else if (!strcmp(k, "--sky")) sky = v;
        else if (!strcmp(k, "--shader")) shader = v;
        else if (!strcmp(k, "--ffmpeg")) ffmpeg = v;
        else {
            fprintf(stderr, "unknown option %s\n", k);
            return 1;
        }
    }
    if (width < 2 || height < 2 || width % 2 || height % 2) die("--width and --height must be even");
    if (fabs(a) >= M) die("--a must be smaller than --M (otherwise there's no horizon)");
    int png = ends_with(out, ".png");
    int frames = png ? 1 : (int)lround(seconds * fps);
    if (frames < 1) die("no frames to draw");

    char shader_path[4096];
    if (!shader) {
        const char *slash = strrchr(argv[0], '/');
        int dirlen = slash ? (int)(slash - argv[0] + 1) : 0;
        snprintf(shader_path, sizeof shader_path, "%.*sblackhole.comp", dirlen, argv[0]);
        shader = shader_path;
    }

    create_context();
    prog = build_program(shader);
    glUseProgram(prog);
    load_sky(sky);
    loc_camera = U("camera");
    loc_time = U("time");
    loc_row = U("rowOffset");
    loc_escape = U("rEscape");

    Settings S = {.M = M, .a = a, .fov = fov, .step = step, .tol = tol, .rk_tol = rk_tol,
                  .rk_max_iter = rk_max_iter, .max_steps = max_steps, .disk = disk,
                  .doppler = doppler};
    if (serving) return serve(S);

    make_target(width, height);
    apply_settings(&S, width, height, 1);
    if (glGetError() != 0) die("OpenGL setup failed");

    char cmd[8192];
    if (png)
        snprintf(cmd, sizeof cmd,
                 "'%s' -loglevel error -y -f rawvideo -pix_fmt rgba -s %dx%d -i - "
                 "-frames:v 1 -update 1 '%s'", ffmpeg, width, height, out);
    else
        snprintf(cmd, sizeof cmd,
                 "'%s' -loglevel error -y -f rawvideo -pix_fmt rgba -s %dx%d -r %g -i - "
                 "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart '%s'",
                 ffmpeg, width, height, fps, out);
    W.pipe = popen(cmd, "w");
    if (!W.pipe) die("couldn't start ffmpeg");
    W.frames = frames;
    W.bytes = (size_t)width * height * 4;
    W.buf[0] = malloc(W.bytes);
    W.buf[1] = malloc(W.bytes);
    if (!W.buf[0] || !W.buf[1]) die("out of memory");
    pthread_t writer_thread;
    pthread_create(&writer_thread, NULL, writer, NULL);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    double total_render = 0, start = now();
    for (int f = 0; f < frames; f++) {
        int index = png ? frame_only : f;
        double t = index / fps;
        double cam[3];
        /* frames / fps is one full orbit, so the video loops */
        flythrough(png ? t / seconds : (double)f / frames, distance * M, cam);
        double dt = draw(cam, t * disk_speed * M);
        total_render += dt;

        int slot = f % 2;
        pthread_mutex_lock(&W.lock);
        while (W.full[slot]) pthread_cond_wait(&W.changed, &W.lock);
        pthread_mutex_unlock(&W.lock);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, W.buf[slot]);
        pthread_mutex_lock(&W.lock);
        W.full[slot] = 1;
        pthread_cond_signal(&W.changed);
        pthread_mutex_unlock(&W.lock);
        fprintf(stderr, "\rframe %d/%d  %.1f ms", f + 1, frames, 1e3 * dt);
    }
    pthread_join(writer_thread, NULL);
    if (W.failed) die("writing to ffmpeg failed");
    if (pclose(W.pipe) != 0) die("ffmpeg failed");
    double wall = now() - start;
    fprintf(stderr, "\nwrote %s\n", out);
    fprintf(stderr, "%d frames: %.2f s rendering (%.1f frames/s), %.2f s in all (%.1f frames/s)\n",
            frames, total_render, frames / total_render, wall, frames / wall);
    if (!png)
        fprintf(stderr, "the video is %.2f s long, so rendering ran at %.2fx real time\n",
                frames / fps, (frames / fps) / wall);
    return 0;
}
