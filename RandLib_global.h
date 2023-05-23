#ifndef RANDLIB_GLOBAL_H
#define RANDLIB_GLOBAL_H

#if defined(_WIN32) || defined(_WIN64)
#define Q_DECL_EXPORT __declspec(dllexport)
#define Q_DECL_IMPORT __declspec(dllimport)
#else
#define Q_DECL_EXPORT __attribute__((visibility("default")))
#define Q_DECL_IMPORT __attribute__((visibility("default")))
#endif

#if RANDLIB_LIBRARY
#define RANDLIBSHARED_EXPORT Q_DECL_EXPORT
#else
#define RANDLIBSHARED_EXPORT Q_DECL_IMPORT
#endif

enum RANDLIBSHARED_EXPORT SUPPORT_TYPE {
  FINITE_T,
  RIGHTSEMIFINITE_T,
  LEFTSEMIFINITE_T,
  INFINITE_T
};

#endif // RANDLIB_GLOBAL_H
