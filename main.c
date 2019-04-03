// Created By Tzachi Sheratzky //

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> 
#include <stdarg.h>
#include "GL/glut.h" 
#include "glm.h"
#include "MatrixAlgebraLib.h"

//////////////////////////////////////////////////////////////////////
// Grphics Pipeline section
//////////////////////////////////////////////////////////////////////
#define WIN_SIZE 500
#define CAMERA_DISTANCE_FROM_AXIS_CENTER 10

typedef struct {
	GLfloat point3D[4];
	GLfloat normal[4];
	GLfloat point3DeyeCoordinates[4];
	GLfloat NormalEyeCoordinates[4];
	GLfloat pointScreen[4];
	GLfloat PixelValue;
} Vertex;

enum ProjectionTypeEnum { ORTHOGRAPHIC = 1, PERSPECTIVE };
enum DisplayTypeEnum { WIREFRAME = 11, LIGHTING_FLAT, LIGHTING_GOURARD };
enum DisplayNormalEnum { DISPLAY_NORMAL_YES = 21, DISPLAY_NORMAL_NO };

GLfloat LookAtMat[16];
GLfloat TranslateMat[16];
GLfloat ScaleMat[16];
GLfloat OrthoMat[16];
GLfloat PerspMat[16];
GLfloat ViewportMat[16];

typedef struct {
	GLfloat ModelMinVec[3]; //(left, bottom, near) of a model.
	GLfloat ModelMaxVec[3]; //(right, top, far) of a model.
	GLfloat CameraPos[3];
	GLfloat ModelScale;
	GLfloat ModelTranslateVector[3];
	enum ProjectionTypeEnum ProjectionType;
	enum DisplayTypeEnum DisplayType;
	enum DisplayNormalEnum DisplayNormals;
	GLfloat Lighting_Diffuse;
	GLfloat Lighting_Specular;
	GLfloat Lighting_Ambient;
	GLfloat Lighting_sHininess;
	GLfloat LightPosition[3];
} GuiParamsForYou;

GuiParamsForYou GlobalGuiParamsForYou;

//written for you
void setPixel(GLint x, GLint y, GLfloat r, GLfloat g, GLfloat b);

//you should write
void ModelProcessing();
void VertexProcessing(Vertex *v);
void FaceProcessing(Vertex *v1, Vertex *v2, Vertex *v3, GLfloat FaceNormal[3]);
GLfloat LightingEquation(GLfloat point[3], GLfloat PointNormal[3], GLfloat LightPos[3], GLfloat Kd, GLfloat Ks, GLfloat Ka, GLfloat n);
void DrawLineBresenham(GLint x1, GLint y1, GLint x2, GLint y2, GLfloat r, GLfloat g, GLfloat b);
void calcTranslateMat(GLfloat x, GLfloat y, GLfloat z);
void calcScaleMat(GLfloat scale);
void calcLookAtMat(GLfloat V[]);
void calcOrthoMat(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far);
void calcPerspectiveMat(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far);
void calcViewportMat(GLfloat currentX, GLfloat currentY, GLfloat windowWidth, GLfloat windowHeight);
GLfloat findMax(GLfloat firstNum, GLfloat secondNum, GLfloat thirdNum);

GLMmodel *model_ptr;
void ClearColorBuffer();
void DisplayColorBuffer();

void GraphicsPipeline()
{
	static GLuint i;
	static GLMgroup* group;
	static GLMtriangle* triangle;
	Vertex v1, v2, v3;
	GLfloat FaceNormal[3];

	ClearColorBuffer();

	//calling ModelProcessing every time refreshing screen
	ModelProcessing();

	//call VertexProcessing for every vertrx
	//and then call FaceProcessing for every face
	group = model_ptr->groups;
	while (group) {
		for (i = 0; i < group->numtriangles; i++) {
			triangle = &(model_ptr->triangles[group->triangles[i]]);

			MatrixCopy(v1.point3D, &model_ptr->vertices[3 * triangle->vindices[0]], 3);
			v1.point3D[3] = 1;
			MatrixCopy(v1.normal, &model_ptr->normals[3 * triangle->nindices[0]], 3);
			v1.normal[3] = 1;
			VertexProcessing(&v1);

			MatrixCopy(v2.point3D, &model_ptr->vertices[3 * triangle->vindices[1]], 3);
			v2.point3D[3] = 1;
			MatrixCopy(v2.normal, &model_ptr->normals[3 * triangle->nindices[1]], 3);
			v2.normal[3] = 1;
			VertexProcessing(&v2);

			MatrixCopy(v3.point3D, &model_ptr->vertices[3 * triangle->vindices[2]], 3);
			v3.point3D[3] = 1;
			MatrixCopy(v3.normal, &model_ptr->normals[3 * triangle->nindices[2]], 3);
			v3.normal[3] = 1;
			VertexProcessing(&v3);

			MatrixCopy(FaceNormal, &model_ptr->facetnorms[3 * triangle->findex], 3);

			FaceProcessing(&v1, &v2, &v3, FaceNormal);
		}
		group = group->next;
	}

	DisplayColorBuffer();
}

GLfloat Zbuffer[WIN_SIZE][WIN_SIZE];
GLfloat Mmodeling[16];
GLfloat Mlookat[16];
GLfloat Mprojection[16];
GLfloat Mviewport[16];

GLfloat findMax(GLfloat deltaX, GLfloat deltaY, GLfloat deltaZ) {

	if (deltaX >= deltaY && deltaX >= deltaZ)
		return deltaX;

	if (deltaY >= deltaX && deltaY >= deltaZ)
		return deltaY;

	if (deltaZ >= deltaX && deltaZ >= deltaY)
		return deltaZ;
}

void ModelProcessing()
{
	GLfloat TempMat[16];
	GLfloat deltaX, deltaY, deltaZ, avgX, avgY, avgZ, max;

	avgX = (GlobalGuiParamsForYou.ModelMaxVec[0] + GlobalGuiParamsForYou.ModelMinVec[0]) / 2.0;
	avgY = (GlobalGuiParamsForYou.ModelMaxVec[1] + GlobalGuiParamsForYou.ModelMinVec[1]) / 2.0;
	avgZ = (GlobalGuiParamsForYou.ModelMaxVec[2] + GlobalGuiParamsForYou.ModelMinVec[2]) / 2.0;

	deltaX = fabs(GlobalGuiParamsForYou.ModelMaxVec[0] - GlobalGuiParamsForYou.ModelMinVec[0]);
	deltaY = fabs(GlobalGuiParamsForYou.ModelMaxVec[1] - GlobalGuiParamsForYou.ModelMinVec[1]);
	deltaZ = fabs(GlobalGuiParamsForYou.ModelMaxVec[2] - GlobalGuiParamsForYou.ModelMinVec[2]);

	//Reshut:
	calcTranslateMat(-avgX, -avgY, -avgZ);

	max = findMax(deltaX, deltaY, deltaZ);

	calcScaleMat(2.0 / max);

	M4multiplyM4(Mmodeling, ScaleMat, TranslateMat);

	// ex2-3: calculating translate transformation matrix
	//////////////////////////////////////////////////////////////////////////////////

	calcTranslateMat(GlobalGuiParamsForYou.ModelTranslateVector[0], GlobalGuiParamsForYou.ModelTranslateVector[1], GlobalGuiParamsForYou.ModelTranslateVector[2]);

	// ex2-3: calculating scale transformation matrix
	//////////////////////////////////////////////////////////////////////////////////
	calcScaleMat(GlobalGuiParamsForYou.ModelScale);

	M4multiplyM4(TempMat, Mmodeling, ScaleMat);
	M4multiplyM4(Mmodeling, TempMat, TranslateMat);

	// ex2-4: calculating lookat transformation matrix
	//////////////////////////////////////////////////////////////////////////////////
	GLfloat V[] = { GlobalGuiParamsForYou.CameraPos[0], GlobalGuiParamsForYou.CameraPos[1], GlobalGuiParamsForYou.CameraPos[2] };
	calcLookAtMat(V);

	// ex2-2: calculating Orthographic or Perspective projection transformation matrix
	//////////////////////////////////////////////////////////////////////////////////
	if (GlobalGuiParamsForYou.ProjectionType == ORTHOGRAPHIC)
	{
		//calcOrthoMat(GlobalGuiParamsForYou.ModelMinVec[0], GlobalGuiParamsForYou.ModelMaxVec[0], GlobalGuiParamsForYou.ModelMinVec[1], GlobalGuiParamsForYou.ModelMaxVec[1], CAMERA_DISTANCE_FROM_AXIS_CENTER-1, CAMERA_DISTANCE_FROM_AXIS_CENTER+1);
		calcOrthoMat(-1, 1, -1, 1, CAMERA_DISTANCE_FROM_AXIS_CENTER - 1, CAMERA_DISTANCE_FROM_AXIS_CENTER + 1);
		MatrixCopy(Mprojection, OrthoMat, 16);
	}
	else if (GlobalGuiParamsForYou.ProjectionType == PERSPECTIVE) {

		calcPerspectiveMat(-2, 2, -2, 2, 19, 20);
		MatrixCopy(Mprojection, PerspMat, 16);
	}


	// ex2-2: calculating viewport transformation matrix
	//////////////////////////////////////////////////////////////////////////////////
	calcViewportMat(0, 0, WIN_SIZE, WIN_SIZE);
	MatrixCopy(Mviewport, ViewportMat, 16);

	// ex3: Z-buffer initialization
	//////////////////////////////////////////////////////////////////////////////////

	for (int x = 0; x < WIN_SIZE; x++)
		for (int y = 0; y < WIN_SIZE; y++)
			Zbuffer[x][y] = 1;
}



void calcTranslateMat(GLfloat x, GLfloat y, GLfloat z)
{
	GLfloat M[16];
	int col = 4;

	M4x4identity(M);

	M[col * 3 + 0] = x;
	M[col * 3 + 1] = y;
	M[col * 3 + 2] = z;

	MatrixCopy(TranslateMat, M, 16);
}

void calcScaleMat(GLfloat scale)
{
	GLfloat M[16];
	int col = 4;

	M4x4identity(M);
	M[col * 0 + 0] = scale;
	M[col * 1 + 1] = scale;
	M[col * 2 + 2] = scale;

	MatrixCopy(ScaleMat, M, 16);
}

void calcLookAtMat(GLfloat V[])
{
	GLfloat M[16];
	GLfloat Center[3] = { 0,0,0 };
	GLfloat w[3] = { V[0] - Center[0], V[1] - Center[1], V[2] - Center[2] };
	GLfloat Up[3] = { 0,1,0 };
	GLfloat u[3], v[3];
	int col = 4;

	M4x4identity(M);
	V3Normalize(w);
	V3cross(u, Up, w);
	V3Normalize(u);
	V3cross(v, w, u);
	V3Normalize(v);

	M[col * 0 + 0] = u[0];
	M[col * 1 + 0] = u[1];
	M[col * 2 + 0] = u[2];

	M[col * 0 + 1] = v[0];
	M[col * 1 + 1] = v[1];
	M[col * 2 + 1] = v[2];

	M[col * 0 + 2] = w[0];
	M[col * 1 + 2] = w[1];
	M[col * 2 + 2] = w[2];

	calcTranslateMat(-V[0], -V[1], -V[2]);
	M4multiplyM4(Mlookat, M, TranslateMat);

}

void calcOrthoMat(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far)
{
	GLfloat M[16];
	int col = 4;

	M4x4identity(M);
	M[col * 0 + 0] = 2 / (right - left);
	M[col * 1 + 1] = 2 / (top - bottom);
	M[col * 2 + 2] = -2 / (far - near);

	M[col * 3 + 0] = -(right + left) / (right - left);
	M[col * 3 + 1] = -(top + bottom) / (top - bottom);
	M[col * 3 + 2] = -(far + near) / (far - near);

	MatrixCopy(OrthoMat, M, 16);
}

void calcPerspectiveMat(GLfloat left, GLfloat right, GLfloat bottom, GLfloat top, GLfloat near, GLfloat far)
{
	GLfloat M[16];
	int col = 4;

	M4x4identity(M);

	M[col * 0 + 0] = 2 * near / (right - left);
	M[col * 2 + 0] = (right + left) / (right - left);
	M[col * 1 + 1] = 2 * near / (top - bottom);
	M[col * 2 + 1] = (top + bottom) / (top - bottom);
	M[col * 2 + 2] = -(far + near) / (far - near);
	M[col * 3 + 2] = -2 * far*near / (far - near);
	M[col * 2 + 3] = -1;
	M[col * 3 + 3] = 0;

	MatrixCopy(PerspMat, M, 16);
}

void calcViewportMat(GLfloat currentX, GLfloat currentY, GLfloat windowWidth, GLfloat windowHeight)
{
	GLfloat MRes[16];
	GLfloat MTranslate[16];
	GLfloat MScale[16];
	int col = 4;

	M4x4identity(MTranslate);
	M4x4identity(MScale);

	MTranslate[col * 3 + 0] = currentX + (windowWidth / 2.0);
	MTranslate[col * 3 + 1] = currentY + (windowHeight / 2.0);
	MTranslate[col * 3 + 2] = 0.5;

	MScale[col * 0 + 0] = windowWidth / 2.0;
	MScale[col * 1 + 1] = windowHeight / 2.0;
	MScale[col * 2 + 2] = 0.5;

	M4multiplyM4(MRes, MTranslate, MScale);

	MatrixCopy(ViewportMat, MRes, 16);
}


void VertexProcessing(Vertex *v)
{
	GLfloat point3DafterModelingTrans[4];
	GLfloat temp1[4], temp2[4];
	GLfloat point3D_plusNormal_screen[4];
	GLfloat Mmodeling3x3[9], Mlookat3x3[9];

	// ex2-3: modeling transformation v->point3D --> point3DafterModelingTrans
	//////////////////////////////////////////////////////////////////////////////////	
	M4multiplyV4(point3DafterModelingTrans, Mmodeling, v->point3D);

	// ex2-4: lookat transformation point3DafterModelingTrans --> v->point3DeyeCoordinates
	//////////////////////////////////////////////////////////////////////////////////
	M4multiplyV4(v->point3DeyeCoordinates, Mlookat, point3DafterModelingTrans);

	// ex2-2: transformation from eye coordinates to screen coordinates v->point3DeyeCoordinates --> v->pointScreen
	//////////////////////////////////////////////////////////////////////////////////	
	M4multiplyV4(temp1, Mprojection, v->point3DeyeCoordinates);
	M4multiplyV4(v->pointScreen, Mviewport, temp1);

	// ex2-5: transformation normal from object coordinates to eye coordinates v->normal --> v->NormalEyeCoordinates
	//////////////////////////////////////////////////////////////////////////////////
	M3fromM4(Mmodeling3x3, Mmodeling);
	M3fromM4(Mlookat3x3, Mlookat);
	M3multiplyV3(temp1, Mmodeling3x3, v->normal);
	M3multiplyV3(v->NormalEyeCoordinates, Mlookat3x3, temp1);
	V3Normalize(v->NormalEyeCoordinates);
	v->NormalEyeCoordinates[3] = 1;

	// ex2-5: drawing normals 
	//////////////////////////////////////////////////////////////////////////////////
	if (GlobalGuiParamsForYou.DisplayNormals == DISPLAY_NORMAL_YES) {
		V4HomogeneousDivide(v->point3DeyeCoordinates);
		VscalarMultiply(temp1, v->NormalEyeCoordinates, 0.05, 3);
		Vplus(temp2, v->point3DeyeCoordinates, temp1, 4);
		temp2[3] = 1;
		M4multiplyV4(temp1, Mprojection, temp2);
		M4multiplyV4(point3D_plusNormal_screen, Mviewport, temp1);
		V4HomogeneousDivide(point3D_plusNormal_screen);
		V4HomogeneousDivide(v->pointScreen);
		DrawLineBresenham(round(v->pointScreen[0]), round(v->pointScreen[1]), round(point3D_plusNormal_screen[0]), round(point3D_plusNormal_screen[1]), 0, 0, 1);
	}

	// ex3: calculating lighting for vertex
	//////////////////////////////////////////////////////////////////////////////////
	v->PixelValue = LightingEquation(v->point3DeyeCoordinates, v->NormalEyeCoordinates, GlobalGuiParamsForYou.LightPosition, GlobalGuiParamsForYou.Lighting_Diffuse, GlobalGuiParamsForYou.Lighting_Specular, GlobalGuiParamsForYou.Lighting_Ambient, GlobalGuiParamsForYou.Lighting_sHininess);
}

GLfloat getRectangleHeight(Vertex v1, Vertex v2, Vertex v3)
{
	GLfloat Dv1v2 = fabs(v1.pointScreen[1] - v2.pointScreen[1]);
	GLfloat Dv1v3 = fabs(v1.pointScreen[1] - v3.pointScreen[1]);
	GLfloat Dv2v3 = fabs(v2.pointScreen[1] - v3.pointScreen[1]);
	GLfloat retVal;

	if (Dv1v2 >= Dv1v3 && Dv1v2 >= Dv2v3)
		retVal = Dv1v2;
	else if (Dv1v3 >= Dv1v2 && Dv1v3 >= Dv2v3)
		retVal = Dv1v3;
	else if (Dv2v3 >= Dv1v2 && Dv2v3 >= Dv1v3)
		retVal = Dv2v3;

	return retVal;
}

GLfloat getRectangleWidth(Vertex v1, Vertex v2, Vertex v3)
{
	GLfloat Dv1v2 = fabs(v1.pointScreen[0] - v2.pointScreen[0]);
	GLfloat Dv1v3 = fabs(v1.pointScreen[0] - v3.pointScreen[0]);
	GLfloat Dv2v3 = fabs(v2.pointScreen[0] - v3.pointScreen[0]);
	GLfloat retVal;

	if (Dv1v2 >= Dv1v3 && Dv1v2 >= Dv2v3)
		retVal = Dv1v2;
	else if (Dv1v3 >= Dv1v2 && Dv1v3 >= Dv2v3)
		retVal = Dv1v3;
	else if (Dv2v3 >= Dv1v2 && Dv2v3 >= Dv1v3)
		retVal = Dv2v3;

	return retVal;
}

GLfloat getMinX(Vertex v1, Vertex v2, Vertex  v3)
{
	GLfloat retVal;

	if (v1.pointScreen[0] <= v2.pointScreen[0] && v1.pointScreen[0] <= v3.pointScreen[0])
		retVal = v1.pointScreen[0];
	else if (v2.pointScreen[0] <= v1.pointScreen[0] && v2.pointScreen[0] <= v3.pointScreen[0])
		retVal = v2.pointScreen[0];
	else if (v3.pointScreen[0] <= v1.pointScreen[0] && v3.pointScreen[0] <= v2.pointScreen[0])
		retVal = v3.pointScreen[0];

	return retVal;
}

GLfloat getMinY(Vertex v1, Vertex v2, Vertex  v3)
{
	GLfloat retVal;

	if (v1.pointScreen[1] <= v2.pointScreen[1] && v1.pointScreen[1] <= v3.pointScreen[1])
		retVal = v1.pointScreen[1];
	else if (v2.pointScreen[1] <= v1.pointScreen[1] && v2.pointScreen[1] <= v3.pointScreen[1])
		retVal = v2.pointScreen[1];
	else if (v3.pointScreen[1] <= v1.pointScreen[1] && v3.pointScreen[1] <= v2.pointScreen[1])
		retVal = v3.pointScreen[0];

	return retVal;
}

int edgeFunction(int Px, int Py, Vertex v1, Vertex v2)
{
	return ((Px - v1.pointScreen[0]) * (v2.pointScreen[1] - v1.pointScreen[1]) - (Py - v1.pointScreen[1]) * (v2.pointScreen[0] - v1.pointScreen[0]) >= 0);
}


void FaceProcessing(Vertex *v1, Vertex *v2, Vertex *v3, GLfloat FaceNormal[3])
{

	V4HomogeneousDivide(v1->pointScreen);
	V4HomogeneousDivide(v2->pointScreen);
	V4HomogeneousDivide(v3->pointScreen);

	if (GlobalGuiParamsForYou.DisplayType == WIREFRAME)
	{
		DrawLineBresenham(round(v1->pointScreen[0]), round(v1->pointScreen[1]), round(v2->pointScreen[0]), round(v2->pointScreen[1]), 1, 1, 1);
		DrawLineBresenham(round(v2->pointScreen[0]), round(v2->pointScreen[1]), round(v3->pointScreen[0]), round(v3->pointScreen[1]), 1, 1, 1);
		DrawLineBresenham(round(v3->pointScreen[0]), round(v3->pointScreen[1]), round(v1->pointScreen[0]), round(v1->pointScreen[1]), 1, 1, 1);

	}
	else {
		//ex3: Barycentric Coordinates and lighting
		//////////////////////////////////////////////////////////////////////////////////

		GLfloat Av1v2 = -v1->pointScreen[1] + v2->pointScreen[1]; // -y1+y2
		GLfloat Bv1v2 = -v2->pointScreen[0] + v1->pointScreen[0]; //-x2+x1
		GLfloat Cv1v2 = v1->pointScreen[1] * (-Bv1v2) - v1->pointScreen[0] * Av1v2; //x1*y2-x2*y1		

		GLfloat Av1v3 = -v1->pointScreen[1] + v3->pointScreen[1]; // -y1+y3
		GLfloat Bv1v3 = -v3->pointScreen[0] + v1->pointScreen[0]; // -x3+x1
		GLfloat Cv1v3 = v1->pointScreen[1] * (-Bv1v3) - v1->pointScreen[0] * Av1v3; //x1*y3-x3*y1

		GLfloat Av2v3 = -v2->pointScreen[1] + v3->pointScreen[1]; // -y2+y3
		GLfloat Bv2v3 = -v3->pointScreen[0] + v2->pointScreen[0]; // -x3+x2
		GLfloat Cv2v3 = v2->pointScreen[1] * (-Bv2v3) - v2->pointScreen[0] * Av2v3; //x2*y3-x3*y2

		GLfloat rectangleWidth = getRectangleWidth(*v1, *v2, *v3);
		GLfloat rectangleHeight = getRectangleHeight(*v1, *v2, *v3);

		GLfloat width = getMinX(*v1, *v2, *v3) + rectangleWidth;
		GLfloat height = getMinY(*v1, *v2, *v3) + rectangleHeight;

		// Find bounding box values
		int xMaxTemp = max((v1->pointScreen[0]), (v2->pointScreen[0]));
		int boxMaxX = max(xMaxTemp, (v3->pointScreen[0]));

		int yMaxTemp = max((v1->pointScreen[1]), (v2->pointScreen[1]));
		int boxMaxY = max(yMaxTemp, (v3->pointScreen[1]));

		int xMinTemp = min((v1->pointScreen[0]), (v2->pointScreen[0]));
		int boxMinX = min(xMinTemp, (v3->pointScreen[0]));

		int yMinTemp = min((v1->pointScreen[1]), (v2->pointScreen[1]));
		int boxMinY = min(yMinTemp, (v3->pointScreen[1]));

		for (int Px = boxMinX; Px <= boxMaxX; Px++)
		{
			for (int Py = boxMinY; Py <= boxMaxY; Py++)
			{
				GLfloat alpha = (Av1v2 * Px + Bv1v2 * Py + Cv1v2) / (Av1v2 * v3->pointScreen[0] + Bv1v2 * v3->pointScreen[1] + Cv1v2);
				GLfloat beta = (Av1v3 * Px + Bv1v3 * Py + Cv1v3) / (Av1v3 * v2->pointScreen[0] + Bv1v3 * v2->pointScreen[1] + Cv1v3);
				GLfloat gama = (Av2v3 * Px + Bv2v3 * Py + Cv2v3) / (Av2v3 * v1->pointScreen[0] + Bv2v3 * v1->pointScreen[1] + Cv2v3);

				if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && gama >= 0 && gama <= 1) {

					GLfloat light;

					switch (GlobalGuiParamsForYou.DisplayType)
					{
					case LIGHTING_FLAT:
						light = (v1->PixelValue + v2->PixelValue + v3->PixelValue) / 3.0;
						break;
					case LIGHTING_GOURARD:
						light = v1->PixelValue * gama + v2->PixelValue * beta + v3->PixelValue * alpha;
					default:
						break;
					}

					GLfloat oneOverZ = v1->pointScreen[2] * gama + v2->pointScreen[2] * beta + v3->pointScreen[2] * alpha;

					if (Px >= 0 && Px < WIN_SIZE && Py < WIN_SIZE && Py >= 0) {

						if (oneOverZ < Zbuffer[Px][Py]) {
							Zbuffer[Px][Py] = oneOverZ;
							setPixel(Px, Py, light, light, light);
						}

					}
				}
			}

		}
	}
}



void DrawLineBresenham(GLint x1, GLint y1, GLint x2, GLint y2, GLfloat r, GLfloat g, GLfloat b)
{
	//ex2.1: implement Bresenham line drawing algorithm
	//////////////////////////////////////////////////////////////////////////////////
	GLint dx, dy, diff;
	GLint xINC, yINC;
	GLint x, y;

	dx = x2 - x1;
	dy = y2 - y1;

	xINC = yINC = 1;

	if (dx < 0) dx = -dx;
	if (dy < 0) dy = -dy;

	if (x2 < x1) xINC = -1;
	if (y2 < y1) yINC = -1;

	x = x1; y = y1;

	if (dx > dy) {

		setPixel(x, y, 1, 1, 1);
		diff = 2 * dy - dx;

		for (int i = 0; i < dx; i++) {
			if (diff >= 0) {
				y += yINC;
				diff += 2 * (dy - dx);
			}
			else {
				diff += 2 * dy;
			}

			x += xINC;
			setPixel(x, y, 1, 1, 1);
		}

	}
	else if (dy > dx) {

		setPixel(x, y, 1, 1, 1);
		diff = 2 * dx - dy;

		for (int i = 0; i < dy; i++) {
			if (diff >= 0) {
				x += xINC;
				diff += 2 * (dx - dy);
			}
			else {
				diff += 2 * dx;
			}

			y += yINC;
			setPixel(x, y, 1, 1, 1);
		}
	}

}

GLfloat LightingEquation(GLfloat point[3], GLfloat PointNormal[3], GLfloat LightPos[3], GLfloat Kd, GLfloat Ks, GLfloat Ka, GLfloat n)
{
	//ex3: calculate lighting equation
	//////////////////////////////////////////////////////////////////////////////////
	GLfloat L[3] = { LightPos[0] - point[0], LightPos[1] - point[1], LightPos[2] - point[2] };
	V3Normalize(L);
	GLfloat V[3] = { -point[0] , -point[1] , -point[2] };
	V3Normalize(V);
	GLfloat Res[3];
	VscalarMultiply(Res, PointNormal, 2.0*V3dot(PointNormal, L), 3);
	GLfloat Reflection[3];
	Vminus(Reflection, Res, L, 3);
	V3Normalize(Reflection);

	GLfloat diffuseTerm = Kd * max(V3dot(PointNormal, L), 0);

	GLfloat specularTerm = Ks * pow(max(V3dot(Reflection, V), 0), n);

	GLfloat ambientTerm = Ka;

	return diffuseTerm + specularTerm + ambientTerm;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// GUI section
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

//function declerations
void InitGuiGlobalParams();
void drawingCB(void);
void reshapeCB(int width, int height);
void keyboardCB(unsigned char key, int x, int y);
void keyboardSpecialCB(int key, int x, int y);
void MouseClickCB(int button, int state, int x, int y);
void MouseMotionCB(int x, int y);
void menuCB(int value);
void TerminationErrorFunc(char *ErrorString);
void LoadModelFile();
void DisplayColorBuffer();
void drawstr(char* FontName, int FontSize, GLuint x, GLuint y, char* format, ...);
void TerminationErrorFunc(char *ErrorString);

enum FileNumberEnum { TEAPOT = 100, TEDDY, PUMPKIN, COW, SIMPLE_PYRAMID, FIRST_EXAMPLE, SIMPLE_3D_EXAMPLE, SPHERE, TRIANGLE };

typedef struct {
	enum FileNumberEnum FileNum;
	GLfloat CameraRaduis;
	GLint   CameraAnleHorizontal;
	GLint   CameraAnleVertical;
	GLint   MouseLastPos[2];
} GuiCalculations;

GuiCalculations GlobalGuiCalculations;

GLuint ColorBuffer[WIN_SIZE][WIN_SIZE][3];

int main(int argc, char** argv)
{
	GLint submenu1_id, submenu2_id, submenu3_id, submenu4_id;

	//initizlizing GLUT
	glutInit(&argc, argv);

	//initizlizing GUI globals
	InitGuiGlobalParams();

	//initializing window
	glutInitWindowSize(WIN_SIZE, WIN_SIZE);
	glutInitWindowPosition(900, 100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("Computer Graphics HW");

	//registering callbacks
	glutDisplayFunc(drawingCB);
	glutReshapeFunc(reshapeCB);
	glutKeyboardFunc(keyboardCB);
	glutSpecialFunc(keyboardSpecialCB);
	glutMouseFunc(MouseClickCB);
	glutMotionFunc(MouseMotionCB);

	//registering and creating menu
	submenu1_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("open teapot.obj", TEAPOT);
	glutAddMenuEntry("open teddy.obj", TEDDY);
	glutAddMenuEntry("open pumpkin.obj", PUMPKIN);
	glutAddMenuEntry("open cow.obj", COW);
	glutAddMenuEntry("open Simple3Dexample.obj", SIMPLE_3D_EXAMPLE);
	glutAddMenuEntry("open SimplePyramid.obj", SIMPLE_PYRAMID);
	glutAddMenuEntry("open sphere.obj", SPHERE);
	glutAddMenuEntry("open triangle.obj", TRIANGLE);
	glutAddMenuEntry("open FirstExample.obj", FIRST_EXAMPLE);
	submenu2_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("Orthographic", ORTHOGRAPHIC);
	glutAddMenuEntry("Perspective", PERSPECTIVE);
	submenu3_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("Wireframe", WIREFRAME);
	glutAddMenuEntry("Flat", LIGHTING_FLAT);
	glutAddMenuEntry("Gourard", LIGHTING_GOURARD);
	submenu4_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("Yes", DISPLAY_NORMAL_YES);
	glutAddMenuEntry("No", DISPLAY_NORMAL_NO);
	glutCreateMenu(menuCB);
	glutAddSubMenu("Open Model File", submenu1_id);
	glutAddSubMenu("Projection Type", submenu2_id);
	glutAddSubMenu("Display type", submenu3_id);
	glutAddSubMenu("Display Normals", submenu4_id);
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	LoadModelFile();

	//starting main loop
	glutMainLoop();
}

void drawingCB(void)
{
	GLenum er;

	char DisplayString1[200], DisplayString2[200];

	//clearing the background
	glClearColor(0, 0, 0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//initializing modelview transformation matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	GraphicsPipeline();

	glColor3f(0, 1, 0);
	sprintf(DisplayString1, "Scale:%.1f , Translate: (%.1f,%.1f,%.1f), Camera angles:(%d,%d) position:(%.1f,%.1f,%.1f) ", GlobalGuiParamsForYou.ModelScale, GlobalGuiParamsForYou.ModelTranslateVector[0], GlobalGuiParamsForYou.ModelTranslateVector[1], GlobalGuiParamsForYou.ModelTranslateVector[2], GlobalGuiCalculations.CameraAnleHorizontal, GlobalGuiCalculations.CameraAnleVertical, GlobalGuiParamsForYou.CameraPos[0], GlobalGuiParamsForYou.CameraPos[1], GlobalGuiParamsForYou.CameraPos[2]);
	drawstr("helvetica", 12, 15, 25, DisplayString1);
	sprintf(DisplayString2, "Lighting reflection - Diffuse:%1.2f, Specular:%1.2f, Ambient:%1.2f, sHininess:%1.2f", GlobalGuiParamsForYou.Lighting_Diffuse, GlobalGuiParamsForYou.Lighting_Specular, GlobalGuiParamsForYou.Lighting_Ambient, GlobalGuiParamsForYou.Lighting_sHininess);
	drawstr("helvetica", 12, 15, 10, DisplayString2);

	//swapping buffers and displaying
	glutSwapBuffers();

	//check for errors
	er = glGetError();  //get errors. 0 for no error, find the error codes in: https://www.opengl.org/wiki/OpenGL_Error
	if (er) printf("error: %d\n", er);
}


void LoadModelFile()
{
	if (model_ptr) {
		glmDelete(model_ptr);
		model_ptr = 0;
	}

	switch (GlobalGuiCalculations.FileNum) {
	case TEAPOT:
		model_ptr = glmReadOBJ("teapot.obj");
		break;
	case TEDDY:
		model_ptr = glmReadOBJ("teddy.obj");
		break;
	case PUMPKIN:
		model_ptr = glmReadOBJ("pumpkin.obj");
		break;
	case COW:
		model_ptr = glmReadOBJ("cow.obj");
		break;
	case SIMPLE_PYRAMID:
		model_ptr = glmReadOBJ("SimplePyramid.obj");
		break;
	case FIRST_EXAMPLE:
		model_ptr = glmReadOBJ("FirstExample.obj");
		break;
	case SIMPLE_3D_EXAMPLE:
		model_ptr = glmReadOBJ("Simple3Dexample.obj");
		break;
	case SPHERE:
		model_ptr = glmReadOBJ("sphere.obj");
		break;
	case TRIANGLE:
		model_ptr = glmReadOBJ("triangle.obj");
		break;
	default:
		TerminationErrorFunc("File number not valid");
		break;
	}

	if (!model_ptr)
		TerminationErrorFunc("can not load 3D model");
	//glmUnitize(model_ptr);  //"unitize" a model by translating it

	//to the origin and scaling it to fit in a unit cube around
	//the origin
	glmFacetNormals(model_ptr);  //adding facet normals
	glmVertexNormals(model_ptr, 90.0);  //adding vertex normals

	glmBoundingBox(model_ptr, GlobalGuiParamsForYou.ModelMinVec, GlobalGuiParamsForYou.ModelMaxVec);
}

void ClearColorBuffer()
{
	GLuint x, y;
	for (y = 0; y < WIN_SIZE; y++) {
		for (x = 0; x < WIN_SIZE; x++) {
			ColorBuffer[y][x][0] = 0;
			ColorBuffer[y][x][1] = 0;
			ColorBuffer[y][x][2] = 0;
		}
	}
}

void setPixel(GLint x, GLint y, GLfloat r, GLfloat g, GLfloat b)
{
	if (x >= 0 && x < WIN_SIZE && y >= 0 && y < WIN_SIZE) {
		ColorBuffer[y][x][0] = round(r * 255);
		ColorBuffer[y][x][1] = round(g * 255);
		ColorBuffer[y][x][2] = round(b * 255);
	}
}

void DisplayColorBuffer()
{
	GLuint x, y;
	glBegin(GL_POINTS);
	for (y = 0; y < WIN_SIZE; y++) {
		for (x = 0; x < WIN_SIZE; x++) {
			glColor3ub(min(255, ColorBuffer[y][x][0]), min(255, ColorBuffer[y][x][1]), min(255, ColorBuffer[y][x][2]));
			glVertex2f(x + 0.5, y + 0.5);   // The 0.5 is to target pixel
		}
	}
	glEnd();
}


void InitGuiGlobalParams()
{
	GlobalGuiCalculations.FileNum = TEAPOT;
	GlobalGuiCalculations.CameraRaduis = CAMERA_DISTANCE_FROM_AXIS_CENTER;
	GlobalGuiCalculations.CameraAnleHorizontal = 0;
	GlobalGuiCalculations.CameraAnleVertical = 0;

	GlobalGuiParamsForYou.CameraPos[0] = 0;
	GlobalGuiParamsForYou.CameraPos[1] = 0;
	GlobalGuiParamsForYou.CameraPos[2] = GlobalGuiCalculations.CameraRaduis;

	GlobalGuiParamsForYou.ModelScale = 1;

	GlobalGuiParamsForYou.ModelTranslateVector[0] = 0;
	GlobalGuiParamsForYou.ModelTranslateVector[1] = 0;
	GlobalGuiParamsForYou.ModelTranslateVector[2] = 0;
	GlobalGuiParamsForYou.DisplayType = WIREFRAME;
	GlobalGuiParamsForYou.ProjectionType = ORTHOGRAPHIC;
	GlobalGuiParamsForYou.DisplayNormals = DISPLAY_NORMAL_NO;
	GlobalGuiParamsForYou.Lighting_Diffuse = 0.75;
	GlobalGuiParamsForYou.Lighting_Specular = 0.2;
	GlobalGuiParamsForYou.Lighting_Ambient = 0.2;
	GlobalGuiParamsForYou.Lighting_sHininess = 40;
	GlobalGuiParamsForYou.LightPosition[0] = 10;
	GlobalGuiParamsForYou.LightPosition[1] = 5;
	GlobalGuiParamsForYou.LightPosition[2] = 0;

}


void reshapeCB(int width, int height)
{
	if (width != WIN_SIZE || height != WIN_SIZE)
	{
		glutReshapeWindow(WIN_SIZE, WIN_SIZE);
	}

	//update viewport
	glViewport(0, 0, width, height);

	//clear the transformation matrices (load identity)
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//projection
	gluOrtho2D(0, WIN_SIZE, 0, WIN_SIZE);
}


void keyboardCB(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		exit(0);
		break;
	case '=':
		GlobalGuiParamsForYou.ModelScale += 0.1;
		glutPostRedisplay();
		break;
	case '-':
		GlobalGuiParamsForYou.ModelScale -= 0.1;
		glutPostRedisplay();
		break;
	case 'd':
	case 'D':
		GlobalGuiParamsForYou.Lighting_Diffuse += 0.05;
		glutPostRedisplay();
		break;
	case 'c':
	case 'C':
		GlobalGuiParamsForYou.Lighting_Diffuse -= 0.05;
		glutPostRedisplay();
		break;
	case 's':
	case 'S':
		GlobalGuiParamsForYou.Lighting_Specular += 0.05;
		glutPostRedisplay();
		break;
	case 'x':
	case 'X':
		GlobalGuiParamsForYou.Lighting_Specular -= 0.05;
		glutPostRedisplay();
		break;
	case 'a':
	case 'A':
		GlobalGuiParamsForYou.Lighting_Ambient += 0.05;
		glutPostRedisplay();
		break;
	case 'z':
	case 'Z':
		GlobalGuiParamsForYou.Lighting_Ambient -= 0.05;
		glutPostRedisplay();
		break;
	case 'h':
	case 'H':
		GlobalGuiParamsForYou.Lighting_sHininess += 1;
		glutPostRedisplay();
		break;
	case 'n':
	case 'N':
		GlobalGuiParamsForYou.Lighting_sHininess -= 1;
		glutPostRedisplay();
		break;
	default:
		printf("Key not valid (language shold be english)\n");
	}
}


void keyboardSpecialCB(int key, int x, int y)
{
	switch (key) {
	case GLUT_KEY_LEFT:
		GlobalGuiParamsForYou.ModelTranslateVector[0] -= 0.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_RIGHT:
		GlobalGuiParamsForYou.ModelTranslateVector[0] += 0.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_DOWN:
		GlobalGuiParamsForYou.ModelTranslateVector[2] -= 0.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_UP:
		GlobalGuiParamsForYou.ModelTranslateVector[2] += 0.1;
		glutPostRedisplay();
		break;
	}
}


void MouseClickCB(int button, int state, int x, int y)
{
	GlobalGuiCalculations.MouseLastPos[0] = x;
	GlobalGuiCalculations.MouseLastPos[1] = y;
}

void MouseMotionCB(int x, int y)
{
	GlobalGuiCalculations.CameraAnleHorizontal += (x - GlobalGuiCalculations.MouseLastPos[0]) / 40;
	GlobalGuiCalculations.CameraAnleVertical -= (y - GlobalGuiCalculations.MouseLastPos[1]) / 40;

	if (GlobalGuiCalculations.CameraAnleVertical > 30)
		GlobalGuiCalculations.CameraAnleVertical = 30;
	if (GlobalGuiCalculations.CameraAnleVertical < -30)
		GlobalGuiCalculations.CameraAnleVertical = -30;

	GlobalGuiCalculations.CameraAnleHorizontal = (GlobalGuiCalculations.CameraAnleHorizontal + 360) % 360;
	//GlobalGuiCalculations.CameraAnleVertical   = (GlobalGuiCalculations.CameraAnleVertical   + 360) % 360;

	GlobalGuiParamsForYou.CameraPos[0] = GlobalGuiCalculations.CameraRaduis * sin((float)(GlobalGuiCalculations.CameraAnleVertical + 90)*M_PI / 180) * cos((float)(GlobalGuiCalculations.CameraAnleHorizontal + 90)*M_PI / 180);
	GlobalGuiParamsForYou.CameraPos[2] = GlobalGuiCalculations.CameraRaduis * sin((float)(GlobalGuiCalculations.CameraAnleVertical + 90)*M_PI / 180) * sin((float)(GlobalGuiCalculations.CameraAnleHorizontal + 90)*M_PI / 180);
	GlobalGuiParamsForYou.CameraPos[1] = GlobalGuiCalculations.CameraRaduis * cos((float)(GlobalGuiCalculations.CameraAnleVertical + 90)*M_PI / 180);
	glutPostRedisplay();
}

void menuCB(int value)
{
	switch (value) {
	case ORTHOGRAPHIC:
	case PERSPECTIVE:
		GlobalGuiParamsForYou.ProjectionType = value;
		glutPostRedisplay();
		break;
	case WIREFRAME:
	case LIGHTING_FLAT:
	case LIGHTING_GOURARD:
		GlobalGuiParamsForYou.DisplayType = value;
		glutPostRedisplay();
		break;
	case DISPLAY_NORMAL_YES:
	case DISPLAY_NORMAL_NO:
		GlobalGuiParamsForYou.DisplayNormals = value;
		glutPostRedisplay();
		break;
	default:
		GlobalGuiCalculations.FileNum = value;
		LoadModelFile();
		glutPostRedisplay();
	}
}



void drawstr(char* FontName, int FontSize, GLuint x, GLuint y, char* format, ...)
{
	va_list args;
	char buffer[255], *s;

	GLvoid *font_style = GLUT_BITMAP_TIMES_ROMAN_10;

	font_style = GLUT_BITMAP_HELVETICA_10;
	if (strcmp(FontName, "helvetica") == 0) {
		if (FontSize == 12)
			font_style = GLUT_BITMAP_HELVETICA_12;
		else if (FontSize == 18)
			font_style = GLUT_BITMAP_HELVETICA_18;
	}
	else if (strcmp(FontName, "times roman") == 0) {
		font_style = GLUT_BITMAP_TIMES_ROMAN_10;
		if (FontSize == 24)
			font_style = GLUT_BITMAP_TIMES_ROMAN_24;
	}
	else if (strcmp(FontName, "8x13") == 0) {
		font_style = GLUT_BITMAP_8_BY_13;
	}
	else if (strcmp(FontName, "9x15") == 0) {
		font_style = GLUT_BITMAP_9_BY_15;
	}

	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	glRasterPos2i(x, y);
	for (s = buffer; *s; s++)
		glutBitmapCharacter(font_style, *s);
}


void TerminationErrorFunc(char *ErrorString)
{
	char string[256];
	printf(ErrorString);
	fgets(string, 256, stdin);

	exit(0);
}

