/* 
 * $Id: ngmath.h.sed,v 1.3 2008/07/27 04:02:35 haley Exp $
 */
/************************************************************************
*                                                                       *
*                Copyright (C)  2000                                    *
*        University Corporation for Atmospheric Research                *
*                All Rights Reserved                                    *
*                                                                       *
*    The use of this Software is governed by a License Agreement.       *
*                                                                       *
************************************************************************/


/*
 *  This file contains some system includes used by Ngmath functions,
 *  a source for the NGCALLF macro, the function prototypes for all 
 *  user entry points in the Ngmath library, and some specific defines.
 */
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 *  Fortran function macro.  This macro is used to provide the appropriate
 *  system-specific C function name for it to be Fortran callable.
 */

/*
 *  Function prototypes for the natgrid package.
 */

/* was duplicated in nnuhead.h */
void    c_nnseti(char *, int);
void    c_nngeti(char *, int *);
void    c_nnsetr(char *, float);
void    c_nngetr(char *, float *);

/* was duplicated in nnuheads.h */
void    c_nnsetc(char *, char *);
void    c_nngetc(char *, char *);
float   *c_natgrids(int, float [], float [], float [],
                     int, int, float [], float [], int *);

/* was duplicated in nncheads.h */
void    c_nngetslopes(int, int, float *, int *);
void    c_nngetaspects(int, int, float *, int *);
void    c_nnpntinits(int, float [], float [], float []);
void    c_nnpnts(float, float, float *);
void    c_nnpntend();
void    c_nngetwts(int *, int *, float *, float *, float *, float *);

/* was duplicated in nnuheadd.h */
void    c_nnsetrd(char *, double);
void    c_nngetrd(char *, double *);
double  *c_natgridd(int, double [], double [], double [],
                     int, int, double [], double [], int *);

/* was duplicated in nncheadd.h */
void    c_nngetsloped(int, int, double *, int *);
void    c_nngetaspectd(int, int, double *, int *);
void    c_nnpntinitd(int, double [], double [], double []);
void    c_nnpntd(double, double, double *);
void    c_nnpntendd();
void    c_nngetwtsd(int *, int *, double *, double *, double *, double *);

#ifdef  UNICOS
#include <fortran.h>
#define NGstring            _fcd
#define NGCstrToFstr(cstr,len) ((cstr)?_cptofcd((char *)cstr,len):_cptofcd("",0)
)
#define NGFstrToCstr(fstr) (_fcdtocp(fstr))
#define NGFlgclToClgcl(flog)  (_ltob(&flog))
#define NGClgclToFlgcl(clog)  (_btol(clog))
float   *c_natgrids(int, float [], float [], float [],
float   *c_natgrids(int, float [], float [], float [],
float   *c_natgrids(int, float [], float [], float [],
float   *c_natgrids(int, float [], float [], float [],
                     int, int, float [], float [], int *);
                     int, int, float [], float [], int *);
                     int, int, float [], float [], int *);
                     int, int, float [], float [], int *);
#else
#define NGstring            char *
#define NGCstrToFstr(cstr,len) (char *)cstr
#define NGFstrToCstr(fstr) fstr
#define NGFlgclToClgcl(flog)  flog
#define NGClgclToFlgcl(clog)  clog
#endif

#define NGSTRLEN(cstr)      ((cstr)?strlen(cstr):0)
 
