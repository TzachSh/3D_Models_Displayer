//  
// Created by Ran Dror on April 2018
// Copyright (c) Ran Dror. All right reserved.
// Code can not be copied and/or distributed without the express permission of author
//

/////////////////////////////////////////////////
//vector and matrix algebra helper functions
/////////////////////////////////////////////////
//print matrix 
void PrintMatrix(GLfloat m[], GLuint height, GLuint width); //displaying 2 digits
void PrintMatrix4(GLfloat m[], GLuint height, GLuint width); //displaying 4 digits
//matrix vector multiply
void M3multiplyV3(GLfloat vres[3], GLfloat m[9], GLfloat v[3]);
void M4multiplyV4(GLfloat vres[4], GLfloat m[16], GLfloat v[4]);
//matrix matrix multiply
void M4multiplyM4(GLfloat mres[16], GLfloat m1[16], GLfloat m2[16]);
void M3multiplyM3(GLfloat mres[9], GLfloat m1[9], GLfloat m2[9]);
void MatirxMmultiply(GLfloat mres[], GLfloat m1[], GLuint m1height, GLuint m1width, GLfloat m2[], GLuint m2width);
//matrix copy
void MatrixCopy(GLfloat dst[], GLfloat src[], GLuint NumberElements);
void M3fromM4(GLfloat m3[9], GLfloat m4[]);
//init matrix to identity
void M3x3identity(GLfloat m[9]);
void M4x4identity(GLfloat m[16]);
//matrix invert
GLboolean M4x4invert(GLfloat inverse[16], GLfloat src[16]);
//matrix transpose
void M3transpose(GLfloat dst[9], GLfloat src[9]);
void M4transpose(GLfloat dst[16], GLfloat src[16]);
//matrix determinant
GLfloat M3determinant(GLfloat m[9]);
GLfloat M4determinant(GLfloat m[16]);
GLfloat M4minor(GLfloat m[16], GLuint IndexRow, GLuint IndexCol);
//vector normalize
GLfloat V3Normalize(GLfloat v[3]);
//vector homogeneous divide
GLboolean V3HomogeneousDivide(GLfloat v[3]);
GLboolean V4HomogeneousDivide(GLfloat v[4]);
//vector dot product
GLfloat V3dot(GLfloat v1[3], GLfloat v2[3]);
GLfloat V4dot(GLfloat v1[4], GLfloat v2[4]);
//vector cross product
void V3cross(GLfloat vres[3], GLfloat v1[3], GLfloat v2[3]);
//vector plus and minus
void Vplus(GLfloat vres[], GLfloat v1[], GLfloat v2[], GLuint NumberElements);
void Vminus(GLfloat vres[], GLfloat v1[], GLfloat v2[], GLuint NumberElements);
//vector scalar multiply
void VscalarMultiply(GLfloat vres[], GLfloat v[], GLfloat scalar, GLuint NumberElements);
//vector comparison
GLuint IsSameVector(GLfloat v1[], GLfloat v2[], GLfloat epsilon, GLuint NumberElements);





