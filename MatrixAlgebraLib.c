#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdarg.h>
#include"GL/glut.h"
#include"MatrixAlgebraLib.h"

void TerminationErrorFunc(char *ErrorString);

/////////////////////////////////////////////////
//vector and matrix algebra helper functions
/////////////////////////////////////////////////


void PrintMatrix(GLfloat m[], GLuint height, GLuint width)
{
	GLuint i, j;
	for (i = 0; i < height; i++) {
		printf("| ");
		for (j = 0; j < width; j++) {
			printf("%.2f ", m[i + j*height]);
		}
		printf("|\n");
	}
}
void PrintMatrix4(GLfloat m[], GLuint height, GLuint width)
{
	GLuint i, j;
	for (i = 0; i < height; i++) {
		printf("| ");
		for (j = 0; j < width; j++) {
			printf("%.4f ", m[i + j*height]);
		}
		printf("|\n");
	}
}

void M3multiplyV3(GLfloat vres[3], GLfloat m[9], GLfloat v[3])
{
	GLuint i, j;
	if (vres==v)
		TerminationErrorFunc("function M3multiplyV3: vres and v can not be the same vector!");

	for (i = 0; i < 3; i++) {
		vres[i] = 0;
		for (j = 0; j < 3; j++) {
			vres[i] += m[i + 3 * j] * v[j];
		}
	}
}

void M4multiplyV4(GLfloat vres[4], GLfloat m[16], GLfloat v[4])
{
	GLuint i, j;
	if (vres==v)
		TerminationErrorFunc("function M4multiplyV4: vres and v can not be the same vector!");
	for (i = 0; i < 4; i++) {
		vres[i] = 0;
		for (j = 0; j < 4; j++) {
			vres[i] += m[i + 4 * j] * v[j];
		}
	}
}
/*testing:
#define VSIZE 3
GLfloat m[VSIZE*VSIZE];
GLfloat v[VSIZE];
GLfloat vres[VSIZE];
int i;
for (i = 0; i < VSIZE*VSIZE; i++)
m[i] = i;
for (i = 0; i < VSIZE; i++)
v[i] = i;
PrintMatrix(m, VSIZE, VSIZE);
PrintMatrix(v, VSIZE, 1);
M3multiplyV3(vres, m, v);
PrintMatrix(vres, VSIZE, 1);

//matlab
VSIZE=3
m = zeros(VSIZE,VSIZE);
m(:) = 0:(VSIZE^2-1); m
v = (0:(VSIZE-1))'
m*v
*/


void M4multiplyM4(GLfloat mres[16], GLfloat m1[16], GLfloat m2[16])
{
	GLuint i, j, k;
	if ((mres == m1) || (mres == m2))
		TerminationErrorFunc("function M4multiplyM4: mres and m1 or m2 can not be the same matrix!");
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			mres[i+4*j] = 0;
			for (k = 0; k < 4; k++) {
				mres[i+4*j] += m1[i + 4 * k] * m2[k + 4 * j];
			}
		}
	}
}

void M3multiplyM3(GLfloat mres[9], GLfloat m1[9], GLfloat m2[9])
{
	GLuint i, j, k;
	if ((mres == m1) || (mres == m2))
		TerminationErrorFunc("function M3multiplyM3: mres and m1 or m2 can not be the same matrix!");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			mres[i+3*j] = 0;
			for (k = 0; k < 3; k++) {
				mres[i + 3 * j] += m1[i + 3 * k] * m2[k + 3 * j];
			}
		}
	}
}
/*testing:
#define VSIZE 3
GLfloat m1[VSIZE*VSIZE];
GLfloat m2[VSIZE*VSIZE];
GLfloat mres[VSIZE*VSIZE];
int i;
for (i = 0; i < VSIZE*VSIZE; i++)
	m1[i] = i;
for (i = 0; i < VSIZE*VSIZE; i++)
	m2[i] = i;
PrintMatrix(m1, VSIZE, VSIZE);
PrintMatrix(m2, VSIZE, VSIZE);

M3multiplyM3(mres, m1, m2);

PrintMatrix(mres, VSIZE, VSIZE);

//matlab
VSIZE=3
m1 = zeros(VSIZE,VSIZE);
m2 = m1;
m1(:) = 0:(VSIZE^2-1); m1
m2 = m1
m1*m2
*/


void MatirxMmultiply(GLfloat mres[], GLfloat m1[], GLuint m1height, GLuint m1width, GLfloat m2[], GLuint m2width)
{
	GLuint i, j, k;
	if ((mres == m1) || (mres == m2))
		TerminationErrorFunc("function MatirxMmultiply: mres and m1 or m2 can not be the same matrix!");
	for (i = 0; i < m1height; i++) {
		for (j = 0; j < m2width; j++) {
			mres[i + m1height*j] = 0;
			for (k = 0; k < m1width; k++) {
				mres[i + m1height*j] += m1[i + m1height*k] * m2[k + m1width*j];
			}
		}
	}
}
/*testing:
#define M1height 3
#define M1width  4
#define M2width  5

GLfloat m1[M1height*M1width];
GLfloat m2[M1width*M2width];
GLfloat mres[M1height*M2width];
int i;
for (i = 0; i < M1height*M1width; i++)
m1[i] = i;
for (i = 0; i < M1width*M2width; i++)
m2[i] = i;
PrintMatrix(m1, M1height,M1width);
printf("\n");
PrintMatrix(m2, M1width,M2width);

MatirxMmultiply(mres, m1, M1height, M1width, m2, M2width);

printf("\n");
PrintMatrix(mres, M1height,M2width);

//matlab
M1height = 3
M1width =  4
M2width =  5
m1 = zeros(M1height,M1width);
m1(:) = 0:(M1height*M1width-1); m1
m2 = zeros(M1width,M2width);
m2(:) = 0:(M1width*M2width-1); m2
m1*m2
*/

void MatrixCopy(GLfloat dst[], GLfloat src[], GLuint NumberElements)
{
	GLuint i;
	for (i = 0; i < NumberElements; i++) {
		dst[i] = src[i];
	}
}
/*testing:
#define M1height 3
#define M1width  4

GLfloat m[M1height*M1width];
GLfloat mres[M1height*M1width];
int i;
for (i = 0; i < M1height*M1width; i++)
m[i] = i;
PrintMatrix(m, M1height,M1width);
printf("\n");

MatrixCopy(mres, m, M1height*M1width);

PrintMatrix(mres, M1height, M1width);
*/

void M3fromM4(GLfloat m3[9], GLfloat m4[])
{
	GLuint i,j;
	if (m3 == m4)
		TerminationErrorFunc("function M3fromM4: m3 and m4 can not be the same matrix!");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			m3[i + 3 * j] = m4[i + 4 * j];
		}
	}
}


void M3x3identity(GLfloat m[9])
{
	m[0 + 3 * 0] = 1; m[0 + 3 * 1] = 0; m[0 + 3 * 2] = 0;
	m[1 + 3 * 0] = 0; m[1 + 3 * 1] = 1; m[1 + 3 * 2] = 0;
	m[2 + 3 * 0] = 0; m[2 + 3 * 1] = 0; m[2 + 3 * 2] = 1;
}

void M4x4identity(GLfloat m[16])
{
	m[0 + 4 * 0] = 1; m[0 + 4 * 1] = 0; m[0 + 4 * 2] = 0; m[0 + 4 * 3] = 0;
	m[1 + 4 * 0] = 0; m[1 + 4 * 1] = 1; m[1 + 4 * 2] = 0; m[1 + 4 * 3] = 0;
	m[2 + 4 * 0] = 0; m[2 + 4 * 1] = 0; m[2 + 4 * 2] = 1; m[2 + 4 * 3] = 0;
	m[3 + 4 * 0] = 0; m[3 + 4 * 1] = 0; m[3 + 4 * 2] = 0; m[3 + 4 * 3] = 1;
}
/*testing:
#define M1height 4
GLfloat m[M1height*M1height];
M4x4identity(m);
PrintMatrix(m, M1height, M1height);
*/

void identity(GLdouble m[16])
{
	m[0 + 4 * 0] = 1; m[0 + 4 * 1] = 0; m[0 + 4 * 2] = 0; m[0 + 4 * 3] = 0;
	m[1 + 4 * 0] = 0; m[1 + 4 * 1] = 1; m[1 + 4 * 2] = 0; m[1 + 4 * 3] = 0;
	m[2 + 4 * 0] = 0; m[2 + 4 * 1] = 0; m[2 + 4 * 2] = 1; m[2 + 4 * 3] = 0;
	m[3 + 4 * 0] = 0; m[3 + 4 * 1] = 0; m[3 + 4 * 2] = 0; m[3 + 4 * 3] = 1;
}



GLboolean M4x4invert(GLfloat inverse[16], GLfloat src[16])
{
	GLfloat CofactorMatrix[16];
	GLfloat AdjointMatrix[16];
	GLuint i, j;
	GLint sign = 1;
	if (inverse == src)
		TerminationErrorFunc("function M4x4invert: inverse and src can not be the same matrix!");
	for (i = 0; i < 4; i++){
		for (j = 0; j < 4; j++){
			CofactorMatrix[i + 4 * j] = sign*M4minor(src, i, j);
			sign = sign*-1;
		}
		sign = sign*-1;
	}
	M4transpose(AdjointMatrix, CofactorMatrix);
	VscalarMultiply(inverse, AdjointMatrix, 1 / M4determinant(src), 16);
	return 1;
}
/*testing:
#define VSIZE 4
GLfloat m[VSIZE*VSIZE];
GLfloat mres[VSIZE*VSIZE];
int i;
m[0 + 4 * 0] = 1;  m[0 + 4 * 1] = 2; m[0 + 4 * 2] = 0; m[0 + 4 * 3] = 0;
m[1 + 4 * 0] = -1; m[1 + 4 * 1] = 1; m[1 + 4 * 2] = 1; m[1 + 4 * 3] = 0;
m[2 + 4 * 0] = 1;  m[2 + 4 * 1] = 2; m[2 + 4 * 2] = 3; m[2 + 4 * 3] = 0;
m[3 + 4 * 0] = 0;  m[3 + 4 * 1] = 0; m[3 + 4 * 2] = 0; m[3 + 4 * 3] = 1;

PrintMatrix4(m, VSIZE, VSIZE);
printf("\n");

M4x4invert(mres, m);

PrintMatrix4(mres, VSIZE, VSIZE);

//matlab
m = [ 1, 2, 0, 0; ...
-1, 1, 1, 0; ...
1, 2, 3, 0; ...
0, 0, 0, 1];
inv(m)
*/


void M3transpose(GLfloat dst[9], GLfloat src[9])
{
	GLuint i, j;
	if (dst == src)
		TerminationErrorFunc("function M4transpose: dst and src can not be the same matrix!");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			dst[i + 3 * j] = src[j + 3 * i];
		}
	}
}

void M4transpose(GLfloat dst[16], GLfloat src[16])
{
	GLuint i, j;
	if (dst == src)
		TerminationErrorFunc("function M4transpose: dst and src can not be the same matrix!");
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			dst[i + 4 * j] = src[j + 4 * i];
		}
	}
}


GLfloat M3determinant(GLfloat m[9])
{
	return
		+m[0 + 3 * 0] * m[1 + 3 * 1] * m[2 + 3 * 2]
		+ m[0 + 3 * 1] * m[1 + 3 * 2] * m[2 + 3 * 0]
		+ m[0 + 3 * 2] * m[1 + 3 * 0] * m[2 + 3 * 1]
		- m[0 + 3 * 2] * m[1 + 3 * 1] * m[2 + 3 * 0]
		- m[0 + 3 * 1] * m[1 + 3 * 0] * m[2 + 3 * 2]
		- m[0 + 3 * 0] * m[1 + 3 * 2] * m[2 + 3 * 1];
}
/*testing:
GLfloat m[9];
GLfloat res;
m[0 + 3 * 0] = 1; m[0 + 3 * 1] = 4; m[0 + 3 * 2] = 7;
m[1 + 3 * 0] = 2; m[1 + 3 * 1] = 5; m[1 + 3 * 2] = 8;
m[2 + 3 * 0] = 3; m[2 + 3 * 1] = 6; m[2 + 3 * 2] = -9;
res = M3determinant(m);

m = [[1 2 3]',[4 5 6]', [7 8 -9]']
det(m)
*/

GLfloat M4minor(GLfloat m4[16], GLuint IndexRow, GLuint IndexCol)
{
	GLfloat submatrix[9];
	GLuint i, j, m, n;
	m = 0;
	for (i = 0; i < 4; i++){
		if (i != IndexRow){
			n = 0;
			for (j = 0; j < 4; j++){
				if (j != IndexCol){
					submatrix[m + 3 * n] = m4[i + 4 * j];
					n++;
				}
			}
			m++;
		}
	}
	return M3determinant(submatrix);
}
/*testing:
GLfloat m[16];
GLfloat res;
m[0 + 4 * 0] = 1; m[0 + 4 * 1] = 5; m[0 + 4 * 2] =  9; m[0 + 4 * 3] =  13;
m[1 + 4 * 0] = 2; m[1 + 4 * 1] = 6; m[1 + 4 * 2] = 10; m[1 + 4 * 3] =  14;
m[2 + 4 * 0] = 3; m[2 + 4 * 1] = 7; m[2 + 4 * 2] = 11; m[2 + 4 * 3] =  15;
m[3 + 4 * 0] = 4; m[3 + 4 * 1] = 8; m[3 + 4 * 2] = 12; m[3 + 4 * 3] = -16;
res = M4minor(m, 1, 2);

m = [[1 3 4]',[5 7 8]', [13 15 -16]']
det(m)
*/

GLfloat M4determinant(GLfloat m[16])
{
	return
		+m[0 + 4 * 0] * M4minor(m, 0, 0)
		- m[0 + 4 * 1] * M4minor(m, 0, 1)
		+ m[0 + 4 * 2] * M4minor(m, 0, 2)
		- m[0 + 4 * 3] * M4minor(m, 0, 3);
}
/*
GLfloat m[16];
GLfloat res;
m[0 + 4 * 0] = 1; m[0 + 4 * 1] = 5; m[0 + 4 * 2] =   9; m[0 + 4 * 3] =  13;
m[1 + 4 * 0] = 2; m[1 + 4 * 1] = 6; m[1 + 4 * 2] =  10; m[1 + 4 * 3] =  14;
m[2 + 4 * 0] = 3; m[2 + 4 * 1] = 7; m[2 + 4 * 2] = -11; m[2 + 4 * 3] =  15;
m[3 + 4 * 0] = 4; m[3 + 4 * 1] = 8; m[3 + 4 * 2] =  12; m[3 + 4 * 3] = -16;
res = M4determinant(m);

m = [[1 2 3 4]',[5 6 7 8]', [9 10 -11 12]',[13 14 15 -16]']
det(m)
*/


GLfloat V3Normalize(GLfloat v[3])
{
	GLfloat length;

	length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] /= length;
	v[1] /= length;
	v[2] /= length;

	return length;
}
/*testing:
#define VSIZE 3
GLfloat v1[VSIZE];
int i;
for (i = 0; i < VSIZE; i++)
v1[i] = i+1;
PrintMatrix(v1, VSIZE, 1);
printf("\n");
V3Normalize(v1);
PrintMatrix(v1, VSIZE, 1);

//matlab
v1 = 1:3
v1/norm(v1)
*/

GLboolean V3HomogeneousDivide(GLfloat v[3])
{
	if (v[2] == 0)
		return GL_FALSE;
	else{
		v[0] /= v[2];
		v[1] /= v[2];
		v[2] = 1;
		return GL_TRUE;
	}
}

GLboolean V4HomogeneousDivide(GLfloat v[4])
{
	if (v[3] == 0)
		return GL_FALSE;
	else{
		v[0] /= v[3];
		v[1] /= v[3];
		v[2] /= v[3];
		v[3] = 1;
		return GL_TRUE;
	}
}

GLfloat V3dot(GLfloat v1[3], GLfloat v2[3])
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

GLfloat V4dot(GLfloat v1[4], GLfloat v2[4])
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

void V3cross(GLfloat vres[3], GLfloat v1[3], GLfloat v2[3])
{
	if ((vres == v1) || (vres == v2))
		TerminationErrorFunc("function V3cross: vres and v1 or v2 can not be the same vector!");

	vres[0] = v1[1] * v2[2] - v1[2] * v2[1];
	vres[1] = v1[2] * v2[0] - v1[0] * v2[2];
	vres[2] = v1[0] * v2[1] - v1[1] * v2[0];
}
/*testing:
#define VSIZE 3
GLfloat v1[VSIZE];
GLfloat v2[VSIZE];
GLfloat vres[VSIZE];
int i;
for (i = 0; i < VSIZE; i++)
v1[i] = i+1;
for (i = 0; i < VSIZE; i++)
v2[i] = VSIZE-i;
PrintMatrix(v1, VSIZE, 1);
printf("\n");
PrintMatrix(v2, VSIZE, 1);
printf("\n");
V3cross(vres, v1, v2);
PrintMatrix(vres, VSIZE, 1);
//matlab
v1 = 1:3
v2 = 3:-1:1
cross(v1,v2)
*/

void Vplus(GLfloat vres[], GLfloat v1[], GLfloat v2[], GLuint NumberElements)
{
	GLuint i;
	for (i = 0; i < NumberElements; i++) {
		vres[i] = v1[i] + v2[i];
	}
}

void Vminus(GLfloat vres[], GLfloat v1[], GLfloat v2[], GLuint NumberElements)
{
	GLuint i;
	for (i = 0; i < NumberElements; i++) {
		vres[i] = v1[i] - v2[i];
	}
}

void VscalarMultiply(GLfloat vres[], GLfloat v[], GLfloat scalar, GLuint NumberElements)
{
	GLuint i;
	for (i = 0; i < NumberElements; i++) {
		vres[i] = scalar*v[i];
	}
}

GLuint IsSameVector(GLfloat v1[], GLfloat v2[], GLfloat epsilon, GLuint NumberElements)
{
	GLuint i;
	for (i = 0; i < NumberElements; i++) {
		if (v1[i]<v2[i] - epsilon || v1[i]>v2[i] + epsilon)
			return 0;
	}
	return 1;
}












/*
GLboolean M4x4invert(GLfloat inverse[16], GLfloat src[16])
{
	double t;
	int i, j, k, swap;
	GLdouble tmp[4][4];

	if (inverse == src)
		TerminationErrorFunc("function M4x4invert: inverse and src can not be the same matrix!");

	M4x4identity(inverse);

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			tmp[i][j] = src[i * 4 + j];
		}
	}

	for (i = 0; i < 4; i++) {
		// look for largest element in column. 
		swap = i;
		for (j = i + 1; j < 4; j++) {
			if (fabs(tmp[j][i]) > fabs(tmp[i][i])) {
				swap = j;
			}
		}

		if (swap != i) {
			// swap rows. 
			for (k = 0; k < 4; k++) {
				t = tmp[i][k];
				tmp[i][k] = tmp[swap][k];
				tmp[swap][k] = t;

				t = inverse[i * 4 + k];
				inverse[i * 4 + k] = inverse[swap * 4 + k];
				inverse[swap * 4 + k] = t;
			}
		}

		if (tmp[i][i] == 0) {
			// no non-zero pivot.  the matrix is singular, which
			//shouldn't happen.  This means the user gave us a bad
			//matrix.
			return GL_FALSE;
		}

		t = tmp[i][i];
		for (k = 0; k < 4; k++) {
			tmp[i][k] /= t;
			inverse[i * 4 + k] /= t;
		}
		for (j = 0; j < 4; j++) {
			if (j != i) {
				t = tmp[j][i];
				for (k = 0; k < 4; k++) {
					tmp[j][k] -= tmp[i][k] * t;
					inverse[j * 4 + k] -= inverse[i * 4 + k] * t;
				}
			}
		}
	}
	return GL_TRUE;
}
*/
/*testing:
#define VSIZE 4
GLfloat m[VSIZE*VSIZE];
GLfloat mres[VSIZE*VSIZE];
int i;
m[0 + 4 * 0] = 0.1; m[0 + 4 * 1] =  3.9; m[0 + 4 * 2] = -2.6; m[0 + 4 * 3] =   0;
m[1 + 4 * 0] = 0.2; m[1 + 4 * 1] =    0; m[1 + 4 * 2] = 13.0; m[1 + 4 * 3] =   0;
m[2 + 4 * 0] = 0.3; m[2 + 4 * 1] = -1.3; m[2 + 4 * 2] = -7.8; m[2 + 4 * 3] =   0;
m[3 + 4 * 0] =   0; m[3 + 4 * 1] =    0; m[3 + 4 * 2] =    0; m[3 + 4 * 3] = 0.1;
PrintMatrix4(m, VSIZE, VSIZE);
printf("\n");

M4x4invert(mres, m);

PrintMatrix4(mres, VSIZE, VSIZE);

//matlab
m = zeros(4);
m =[0.1, 3.9, -2.6, 0; ...
0.2,   0, 13.0, 0; ...
0.3,-1.3, -7.8, 0; ...
0,   0,    0, 0.1];
inv(m)
*/
