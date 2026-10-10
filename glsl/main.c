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

int main(int argc, char **argv) {
    double M = 1.0, a = 0.5, h = 1.0;
    long steps = 15000, chunk = 100;
    const char *out = "geodesic.npy", *shader = NULL, *ic_file = NULL;
    double *ics = NULL;
    long n_ic = 0;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(k, "-h") || !strcmp(k, "--help")) {
            fprintf(stderr, "see the comment at the top of main.c for options\n");
            return 0;
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
