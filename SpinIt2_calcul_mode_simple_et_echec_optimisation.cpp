#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <ostream>
#include <igl/readOFF.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <math.h>

#include <igl/gaussian_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>

#include "HalfedgeBuilder.cpp"
#include <list>
#include <queue>
#include <iostream>
#include <vector>

#include <igl/active_set.h>
#include <igl/boundary_facets.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <Eigen/Sparse>



using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;
#define MAX 22

struct Point {
	int x;
	int y;
	int z;
};

struct Spacerepresentation {
	double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
	int Space[MAX][MAX][MAX]; //0 Void 1 Surface 2 Interior
	vector<Point> Cellint;
	vector<Point> Cellbound;
};

typedef struct Spacerepresentation Spacerepresentation;

void Show(Spacerepresentation S);

MatrixXd V;
MatrixXi F;

MatrixXd N_faces;   //computed calling pre-defined functions of LibiGL
MatrixXd N_vertices; //computed calling pre-defined functions of LibiGL

MatrixXd lib_N_vertices;  //computed using face-vertex structure of LibiGL
MatrixXi lib_Deg_vertices;//computed using face-vertex structure of LibiGL

MatrixXd he_N_vertices; //computed using the HalfEdge data structure

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
	switch(key){
		case '1':
			viewer.data().set_normals(N_faces);
			return true;
		case '2':
			viewer.data().set_normals(N_vertices);
			return true;
		case '3':
			viewer.data().set_normals(lib_N_vertices);
			return true;
		case '4':
			viewer.data().set_normals(he_N_vertices);
			return true;
		default: break;
	}
	return false;
}

/**
* Return the degree of a given vertex 'v' by turning around
**/
int vertexDegree(HalfedgeDS he, int v) {
	int edgeinitial=he.getEdge(v);
	int edge = edgeinitial;
	int degree = 0;
	do {
		edge = he.getNext(edge);
		edge = he.getOpposite(edge);
		degree++;
	} while (edge != edgeinitial);
 	return degree;
}

/**
* Compute the vertex normals (he)
**/
void vertexNormals(HalfedgeDS he) {
	// We could compute the normals to the triangle mesh but I'll do that for lib_vertexNormals
	// Instead, I will look at all the edges around V. Like in TD5, we can consider that they are the NN and build a plane. 
	// By coputing to covariance matrix and finding the smallest Eigenvalue, we will get the direction of the normal.

	int nb_points = he.sizeOfVertices();
	he_N_vertices = MatrixXd(nb_points, 3);
	MatrixXd A(1, 3);
	for (int i = 0;i < nb_points;i++) {
		// We know work for Vertice number i
		MatrixXd C = MatrixXd::Zero(3, 3);

		// Compute Pmean for Vertice number i by finding all the vertices connected (same as to find the degree)
		MatrixXd Pmean = MatrixXd::Zero(1, 3);
		//Vector3d Pmean;
		//Pmean[0] = Pmean[1] = Pmean[2];
		int edgeinitial = he.getEdge(i);
		int edge = edgeinitial;
		int degree = 0;
		do {
			edge = he.getNext(edge);
			Pmean = Pmean + V.row(he.getTarget(edge));
			edge = he.getOpposite(edge);
			degree++;
		} while (edge != edgeinitial);
		Pmean = Pmean / degree;

		// Compute covariance Matric for Point i : deuxième tour !
		do {
			edge = he.getNext(edge);
			A = (V.row(he.getTarget(edge)) - Pmean);
			C = C + A.transpose() * A; // We should divide by k but it is unuseful
			edge = he.getOpposite(edge);
		} while (edge != edgeinitial);

		// The normal is in the direction of the smallest Eigen Vector
		EigenSolver<MatrixXd> Diag(C);
		MatrixXcd ev = Diag.eigenvalues();

		//Doc Eigen:The eigenvalues are repeated according to their algebraic multiplicity, so there are as many eigenvalues as rows in the matrix. The eigenvalues are not sorted in any particular order.
		int valmin; // So we have to decide which Eigenvalue we take
		double val0 = ev(0, 0).real();
		double val1 = ev(1, 0).real();
		double val2 = ev(2, 0).real();

		if (val0 <= val1 && val0 <= val2) valmin = 0;
		else if (val1 <= val2) valmin = 1;
		else valmin = 2;

		MatrixXcd Veigen = Diag.eigenvectors().col(valmin);

		he_N_vertices.row(i) = Veigen.col(0).real();
		if (V(i, 0) * he_N_vertices(i, 0) < 0) he_N_vertices.row(i) = -he_N_vertices.row(i); // If else half of the cat is black but I still have black regions... I can see them if I go "inside the cat"
	}
}

void lib_vertexNormals() {
	// I will approximate the problem and compute the normal to the faces : by computng the cross product between the vectors of two edges of the face.
	int nbfaces = F.rows();
	lib_N_vertices = MatrixXd(nbfaces, 3);
	for (int i = 0;i < nbfaces;i++) {
		Vector3d edge1, edge2;
		edge1 = V.row(F(i, 0)) - V.row(F(i, 1));
		edge2 = V.row(F(i, 0)) - V.row(F(i, 2));
		lib_N_vertices.row(i) = edge1.cross(edge2).normalized();
	}
}

Array<double, 3, 2> compute_min_max_on_axes(MatrixXd V) //Xmin, Xmax, Ymin, Ymax, Zmin, Zmax
{
	int n = V.rows();
	Array<double,3,2> Answer;
	
	Answer(0,0) = Answer(1,0) = Answer(2,0) = +100000.0;
	Answer(0,1) = Answer(1,1) = Answer(2,1) = -100000.0;

	for (int i = 0;i < n;i++) {
		if (V(i, 0) < Answer(0,0)) Answer(0,0) = V(i, 0);
		if (V(i, 0) > Answer(0,1)) Answer(0,1) = V(i, 0);
		if (V(i, 1) < Answer(1,0)) Answer(1,0) = V(i, 1);
		if (V(i, 1) > Answer(1,1)) Answer(1,1) = V(i, 1);
		if (V(i, 2) < Answer(2,0)) Answer(2,0) = V(i, 2);
		if (V(i, 2) > Answer(2,1)) Answer(2,1) = V(i, 2);
	}
	return Answer;
}

void translation(MatrixXd &V, Vector3d T)
{
	for (int i = 0;i < V.rows();i++) {
		V(i, 0) -= T(0);
		V(i, 1) -= T(1);
		V(i, 2) -= T(2);
	}
}


void rotation(MatrixXd& V, Matrix3d T)
{
	V = V * T;
}

Vector3d barre(Vector3d a) {
	Vector3d abar;
	abar(0) = a(1);    abar(1) = a(2);   abar(2) = a(0);
	return abar;
}

Vector3d mult_term_by_term(Vector3d a, Vector3d b) {
	Vector3d mult;
	mult(0) = a(0) * b(0);
	mult(1) = a(1) * b(1);
	mult(2) = a(2) * b(2);
	return mult;
}

void Show(Spacerepresentation S) {
	for (int z = 0;z < MAX;z++) {
		cout << "Couche z=" << z << endl;
		for (int i = 0; i < MAX; i++) {
			for (int j = 0;j < MAX;j++) {
				if (S.Space[i][j][z] == 2) cout << ". ";
				else if (S.Space[i][j][z] == 1) cout << "* "; else cout << "  ";
			}
			cout << endl;
		}
		cout << endl;
	}
}


void computeSonsurface(MatrixXd V, MatrixXi F) {

	double s[10];
	for (int i = 0;i < 10;i++) s[i] = 0;
	Vector3d a, b, c, u, v, n, h1, h2, h3, h4, h5, h6, h7, h8, xx;

	for (int i = 0;i < F.rows();i++) {	// For all faces

		
		a = V.row(F(i, 0));
		b = V.row(F(i, 2));
		c = V.row(F(i, 1));
		u = b - a;
		v = c - a;
		u = -u;
		n = -u.cross(v); 
		h1 = a + b + c;
		h2 = mult_term_by_term(a,a) + mult_term_by_term(b,a + b);
		h3 = h2 + mult_term_by_term(c,h1);
		h4 = mult_term_by_term(a, mult_term_by_term(a,a))+ mult_term_by_term(b,h2)+ mult_term_by_term(c,h3);
		h5 = h3 + mult_term_by_term(a,h1 + a);
		h6 = h3 + mult_term_by_term(b,h1 + b);
		h7 = h3 + mult_term_by_term(c,h1 + c);
		h8 = mult_term_by_term(barre(a),h5) + mult_term_by_term(barre(b),h6) + mult_term_by_term(barre(c),h7);

		s[0] += (mult_term_by_term(n,h1))(0);
		xx = mult_term_by_term(n,h3); s[1] += xx(0); s[2] += xx(1); s[3] += xx(2);
		xx = mult_term_by_term(n,h8); s[4] += xx(0); s[5] += xx(1); s[6] += xx(2);
		xx = mult_term_by_term(n,h4); s[7] += xx(0); s[8] += xx(1); s[9] += xx(2);
		//cout << "Resultat des calculs apres la face <<i<< : " << endl;
		//for (int j = 0;j< 10;j++) cout << "Indice " << j << " : " << s[j] << endl;
		//
	}
	s[0] /= 6;
	s[1] /= 24; s[2] /= 24; s[3] /= 24;
	s[4] /= 120; s[5] /= 120; s[6] /= 120;
	s[7] /= 60; s[8] /= 60; s[9] /= 60;
	if (s[0] < 0) for (int i = 0;i < 10;++i) s[i] *= -1;
	// Rho est pris égal à 1 !
	//return s;
	cout << "Resultat des calculs sur la surface : " <<endl;
	for (int i = 0;i < 10;i++) cout<<"Indice "<<i<<" : "<<s[i]<<endl;
}

void createSpaceRep(Spacerepresentation &S,MatrixXd V,MatrixXi F) {
	for (int i = 0;i < MAX;i++)
		for (int j = 0;j < MAX;j++)
			for (int k = 0;k < MAX;k++)
				S.Space[i][j][k] = false;
	double resolution;
	Array<double, 3, 2> XYZminmax = compute_min_max_on_axes(V);
	S.Xmin = XYZminmax(0, 0); S.Xmax = XYZminmax(0, 1); S.Ymin = XYZminmax(1, 0); S.Ymax = XYZminmax(1, 1); S.Zmin = XYZminmax(2, 0); S.Zmax = XYZminmax(2, 1);
	if ((S.Xmax - S.Xmin) > (S.Ymax - S.Ymin) && (S.Xmax - S.Xmin) > (S.Zmax - S.Zmin)) resolution = 2*(MAX - 1)/ (S.Xmax - S.Xmin);
	else if ((S.Ymax - S.Ymin)> (S.Zmax - S.Zmin)) resolution = 2*(MAX - 1) / (S.Ymax - S.Ymin);
	else resolution = 2*(MAX - 1) / (S.Zmax - S.Zmin);
	//cout << "Resolution : " << resolution;
	double x1, y1, z1, x2, y2, z2,x3,y3,z3;
	double a0, b0, c0;
	double avers2, bvers2, cvers2, avers3, bvers3, cvers3;
	double distance12, distance13, distance23, axe2, axe3;
	for (int i = 0;i < F.rows();i++) {
		//cout << "Face n° " << i << endl;
		x1 = V(F(i, 0), 0); y1 = V(F(i, 0), 1); z1 = V(F(i, 0), 2);
		x2 = V(F(i, 1), 0); y2 = V(F(i, 1), 1); z2 = V(F(i, 1), 2);
		x3 = V(F(i, 2), 0); y3 = V(F(i, 2), 1); z3 = V(F(i, 2), 2);

		distance12 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
		distance13 = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3) + (z1 - z3) * (z1 - z3);
		distance23 = (x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3) + (z2 - z3) * (z2 - z3);
		//cout << "x1=" << x1 << "  y1=" << y1 << "  z1=" << z1 << endl;
		//cout << "x2=" << x2 << "  y2=" << y2 << "  z2=" << z2 << endl;
		//cout << "x3=" << x3 << "  y3=" << y3 << "  z3=" << z3 << endl;
		//cout << "Dist12= " << sqrt(distance12) << "  Dist13=" << sqrt(distance13) << "  Dist1=23=" << sqrt(distance23) << endl;

		if (distance23 < distance12 && distance23 < distance13) {// On prend (1,12,13) comme repère       
			a0 = (x1 - S.Xmin) / (S.Xmax - S.Xmin) * (MAX - 1);
			b0 = (y1 - S.Ymin) / (S.Ymax - S.Ymin) * (MAX - 1);
			c0 = (z1 - S.Zmin) / (S.Zmax - S.Zmin) * (MAX - 1);

			avers2 = (x2 - x1) / (S.Xmax - S.Xmin) * (MAX - 1);
			bvers2 = (y2 - y1) / (S.Ymax - S.Ymin) * (MAX - 1);
			cvers2 = (z2 - z1) / (S.Zmax - S.Zmin) * (MAX - 1);

			avers3 = (x3 - x1) / (S.Xmax - S.Xmin) * (MAX - 1);
			bvers3 = (y3 - y1) / (S.Ymax - S.Ymin) * (MAX - 1);
			cvers3 = (z3 - z1) / (S.Zmax - S.Zmin) * (MAX - 1);

			axe2 = sqrt(distance12);
			axe3 = sqrt(distance13);
		}

		else if (distance12 < distance13) { //On prend (3,31,32) comme repère
			a0 = (x3 - S.Xmin) / (S.Xmax - S.Xmin) * (MAX - 1);
			b0 = (y3 - S.Ymin) / (S.Ymax - S.Ymin) * (MAX - 1);
			c0 = (z3 - S.Zmin) / (S.Zmax - S.Zmin) * (MAX - 1);

			avers2 = (x1 - x3) / (S.Xmax - S.Xmin) * (MAX - 1);
			bvers2 = (y1 - y3) / (S.Ymax - S.Ymin) * (MAX - 1);
			cvers2 = (z1 - z3) / (S.Zmax - S.Zmin) * (MAX - 1);

			avers3 = (x2 - x3) / (S.Xmax - S.Xmin) * (MAX - 1);
			bvers3 = (y2 - y3) / (S.Ymax - S.Ymin) * (MAX - 1);
			cvers3 = (z2 - z3) / (S.Zmax - S.Zmin) * (MAX - 1);

			axe2 = sqrt(distance13);
			axe3 = sqrt(distance23);
		}

		else { // On prend (2,21,23) comme repère
			a0 = (x2 - S.Xmin) / (S.Xmax - S.Xmin) * (MAX - 1);
			b0 = (y2 - S.Ymin) / (S.Ymax - S.Ymin) * (MAX - 1);
			c0 = (z2 - S.Zmin) / (S.Zmax - S.Zmin) * (MAX - 1);

			avers2 = (x1 - x2) / (S.Xmax - S.Xmin) * (MAX - 1);
			bvers2 = (y1 - y2) / (S.Ymax - S.Ymin) * (MAX - 1);
			cvers2 = (z1 - z2) / (S.Zmax - S.Zmin) * (MAX - 1);

			avers3 = (x3 - x2) / (S.Xmax - S.Xmin) * (MAX - 1);
			bvers3 = (y3 - y2) / (S.Ymax - S.Ymin) * (MAX - 1);
			cvers3 = (z3 - z2) / (S.Zmax - S.Zmin) * (MAX - 1);

			axe2 = sqrt(distance12);
			axe3 = sqrt(distance23);
		}
		//cout << "a0=" << a0 << "  b0=" << b0 << "  c0=" << c0 << endl;
		//cout << "avers2=" << avers2 << "  bvers2=" << bvers2 << "  cvers2=" << cvers2 << endl;
		//cout << "avers3=" << avers3 << "  bvers3=" << bvers3 << "  cvers3=" << cvers3 << endl;
		//cout << "axe2=" << axe2 << "  axe3=" << axe3 << endl;
		// On a un repère (O,i,j) avec i et j les plus grands côtés et il faut faire toutes les valeurs entières à l'intérieur du triangle
		for (int dir2 = 0;dir2 <= round(resolution*axe2);dir2++)
			for (int dir3 = 0;dir3 <= round(resolution*axe3);dir3++) {

				double xx = a0 + (avers2 * dir2 / axe2 + avers3 * dir3 / axe3)/resolution;
				double yy = b0 + (bvers2 * dir2 / axe2 + bvers3 * dir3 / axe3)/ resolution;
				double zz = c0 + (cvers2 * dir2 / axe2 + cvers3 * dir3 / axe3)/ resolution;
				if ((dir2 / axe2 + dir3 / axe3) <= (resolution+0.5)) {
					if (!S.Space[int(round(xx))][int(round(yy))][int(round(zz))]) {
						S.Space[int(round(xx))][int(round(yy))][int(round(zz))] = 1;
						Point newcell;
						newcell.x = int(round(xx));
						newcell.y = int(round(yy));
						newcell.z = int(round(zz));
						S.Cellbound.push_back(newcell);
						//cout << "Point allume en     " << int(round(xx)) << " - " << int(round(yy)) << " - " << int(round(zz)) << "  -----------------dir2 =" << dir2 << " et dir3= " << dir3 << endl;
					}
				}
		}
		//cout << endl;
	}

	queue<Point> Q;
	Point seed;
	seed.y=seed.z=seed.x = MAX / 2;
	seed.z = MAX / 3;
	Q.push(seed);

	while (!Q.empty()) {
		//cout << Q.size() << " ";
		Point P = Q.front();
		if (!S.Space[P.x][P.y][P.z] && P.x>0 && P.x<MAX-1 && P.y>0 && P.y<MAX-1 && P.z>0 && P.z<MAX-1){
			S.Cellint.push_back(P);
			S.Space[P.x][P.y][P.z] = 2;
			Point Px1, Px2, Py1, Py2, Pz1, Pz2;
			Px1.x = P.x + 1; Px1.y = P.y; Px1.z = P.z;
			Px2.x = P.x - 1; Px2.y = P.y; Px2.z = P.z;
			Py1.x = P.x ; Py1.y = P.y+1 ; Py1.z = P.z;
			Py2.x = P.x ; Py2.y = P.y-1 ; Py2.z = P.z;
			Pz1.x = P.x ; Pz1.y = P.y ; Pz1.z = P.z + 1;
			Pz2.x = P.x ; Pz2.y = P.y ; Pz2.z = P.z - 1;
			Q.push(Px1);
			Q.push(Px2);
			Q.push(Py1);
			Q.push(Py2);
			Q.push(Pz1);
			Q.push(Pz2);
		}
		Q.pop();
	}
}


void computeSonSpaceRep(Spacerepresentation S) {
	double s[10];
	for (int i = 0;i < 10;i++) s[i] = 0;

	double taille_x = (S.Xmax - S.Xmin)  / (MAX-1);
	double taille_y = (S.Ymax - S.Ymin) / (MAX-1);
	double taille_z = (S.Zmax - S.Zmin) / (MAX-1);
	double taille_voxel = taille_x * taille_y * taille_z;
	
	for (int i = 0;i < S.Cellint.size();i++) {
		double xspace = S.Xmin + S.Cellint[i].x * taille_x;
		double yspace = S.Ymin + S.Cellint[i].y * taille_y;
		double zspace = S.Zmin + S.Cellint[i].z * taille_z;
		s[0] += 1;
		s[1] += xspace;
		s[2] += yspace;
		s[3] += zspace;
		s[4] += (xspace * yspace);
		s[5] += (yspace * zspace);
		s[6] += (xspace * zspace);
		s[7] += (xspace * xspace);
		s[8] += (yspace * yspace);
		s[9] += (zspace * zspace);
	}

	for (int i = 0;i < S.Cellbound.size();i++) {
		double xspace = S.Xmin + S.Cellbound[i].x * taille_x;
		double yspace = S.Ymin + S.Cellbound[i].y * taille_y;
		double zspace = S.Zmin + S.Cellbound[i].z * taille_z;
		s[0] += 1;
		s[1] += xspace;
		s[2] += yspace;
		s[3] += zspace;
		s[4] += (xspace * yspace);
		s[5] += (yspace * zspace);
		s[6] += (xspace * zspace);
		s[7] += (xspace * xspace);
		s[8] += (yspace * yspace);
		s[9] += (zspace * zspace);
		}

	// Rho est pris égal à 1 !
	cout << "Resultat des calculs sur le volume : " << endl;

	for (int i = 0;i < 10;i++) {
		s[i] /= taille_voxel;
		cout << "Indice " << i << " : " << s[i] << endl;
	}
}

double compute_energie(VectorXd Beta, Spacerepresentation S, double gammaI, double gammaC) {
	int cellint = Beta.rows();
	double taille_x = (S.Xmax - S.Xmin) / (MAX - 1);
	double taille_y = (S.Ymax - S.Ymin) / (MAX - 1);
	double taille_z = (S.Zmax - S.Zmin) / (MAX - 1);
	double s[10];

	for (int i = 0;i < S.Cellbound.size();i++) {
		double xspace = S.Xmin + S.Cellbound[i].x * taille_x;
		double yspace = S.Ymin + S.Cellbound[i].y * taille_y;
		double zspace = S.Zmin + S.Cellbound[i].z * taille_z;
		s[0] += 1;
		s[1] += xspace;
		s[2] += yspace;
		s[3] += zspace;
		s[4] += (xspace * yspace);
		s[5] += (yspace * zspace);
		s[6] += (xspace * zspace);
		s[7] += (xspace * xspace);
		s[8] += (yspace * yspace);
		s[9] += (zspace * zspace);
	}

	for (int i = 0;i < cellint;i++) {
		double xspace = S.Xmin + S.Cellint[i].x * taille_x;
		double yspace = S.Ymin + S.Cellint[i].y * taille_y;
		double zspace = S.Zmin + S.Cellint[i].z * taille_z;
		s[0] += Beta[i]*1;
		s[1] += Beta[i]*xspace;
		s[2] += Beta[i]*yspace;
		s[3] += Beta[i]*zspace;
		s[4] += Beta[i]*(xspace * yspace);
		s[5] += Beta[i]*(yspace * zspace);
		s[6] += Beta[i]*(xspace * zspace);
		s[7] += Beta[i]*(xspace * xspace);
		s[8] += Beta[i]*(yspace * yspace);
		s[9] += Beta[i]*(zspace * zspace);
	}

	MatrixXd Iplan;
	Iplan.resize(2, 2);
	Iplan << s[8] + s[9], -s[4], -s[4], s[7] + s[9];
	VectorXcd Ixety = Iplan.eigenvalues();
	//double Ia = Ixety(0).real;
	//double Ib = Ixety(1).real;
	//double Ic = s[7] + s[8];
	//double Energie = gammaI * (Ia * Ia + Ib * Ib) / (Ic * Ic) + gammaC * s[3] * s[3];
	// Pas fini attention...
	return 0;
}





// ------------ main program ----------------
int main(int argc, char *argv[]) {

	igl::readOFF("../data/toupie_mod5.off",V,F);
	Array<double, 3, 2> XYZminmax = compute_min_max_on_axes(V);
	double Xmin = XYZminmax(0, 0); double Xmax = XYZminmax(0, 1); double Ymin = XYZminmax(1, 0); double Ymax = XYZminmax(1, 1); double Zmin = XYZminmax(2, 0); double Zmax = XYZminmax(2, 1);
		
		cout << "Avant rotation et translation : " << endl;
	cout << "L'axe x va de " << Xmin << " a " << Xmax << endl;
	cout << "L'axe y va de " << Ymin << " a " << Ymax << endl;
	cout << "L'axe z va de " << Zmin << " a " << Zmax << endl;
		
	Vector3d Trans;

	Trans(0) = (Xmax - Xmin)/2; 	Trans(1) = (Ymax - Ymin) / 2; 	Trans(2) = (Zmax - Zmin) / 2;
	//Trans(0)=0;Trans(1) = 0;Trans(2) = 4;
	//cout << Trans << endl;
	translation(V, Trans);

	Matrix3d Rotation;
	Rotation << 0, 0, 1, 0, 1, 0, 1, 0, 0;
	rotation(V, Rotation);

	Trans(0) = 0;Trans(1) = 0;Trans(2) = (Xmax - Xmin) / 2-18.33;
	translation(V, Trans);

	//Symmétrie sur Z pour avoir Z=0 comme point de contact
	for (int i = 0;i < V.rows();i++) V(i, 2) *= -1;

	XYZminmax = compute_min_max_on_axes(V);
	Xmin = XYZminmax(0, 0);	Xmax = XYZminmax(0, 1); Ymin = XYZminmax(1, 0); Ymax = XYZminmax(1, 1); Zmin = XYZminmax(2, 0); Zmax = XYZminmax(2, 1);
	cout << "Après rotation et translation : " << endl;
	cout << "L'axe x va de " << Xmin << " a " << Xmax << endl;
	cout << "L'axe y va de " << Ymin << " a " << Ymax << endl;
	cout << "L'axe z va de " << Zmin << " a " << Zmax << endl;
	
		double s[10];
	computeSonsurface(V, F);

	Spacerepresentation SpaceRep;
	createSpaceRep(SpaceRep,V,F);
	cout << SpaceRep.Cellbound.size() << "  " << SpaceRep.Cellint.size() << endl;
	Show(SpaceRep);
l : 	computeSonSpaceRep(SpaceRep);

	int cellint = SpaceRep.Cellint.size();
	int cellbound = SpaceRep.Cellbound.size();

	double gammaI = 1;
	double gammaC = 1;

	
	Eigen::VectorXi b;
	Eigen::VectorXd B, bc, lx, ux, Beq, Bieq, Z;
	Eigen::SparseMatrix<double> Q, Aeq, Aieq;


	Eigen::VectorXi nbdevoisins;
	nbdevoisins = VectorXi::Zero(cellint); 

	// Set a Lagrangian over the neigboorings cells
	Eigen::SparseMatrix<double> L;
	Eigen::MatrixXd Lbis;

	L.resize(cellint, cellint);
	Lbis.resize(cellint, cellint);
	for (int i = 1;i < cellint;i++) {
		for (int j = 0;j < i;j++) {
			// Les cellules i et j sont-elles voisines ?
			int distancex = SpaceRep.Cellint[i].x - SpaceRep.Cellint[j].x;
			int distancey = SpaceRep.Cellint[i].y - SpaceRep.Cellint[j].y;
			int distancez = SpaceRep.Cellint[i].z - SpaceRep.Cellint[j].z;
			if (abs(distancex)<=1 && abs(distancey) <= 1 && abs(distancez) <= 1) {
				nbdevoisins(j)++;
				L.insert(i, j) = -1.0;
				L.insert(j, i) = -1.0;
				Lbis(i, j) = Lbis(j, i) = -1.0;
			}
		}
	}
	for (int i = 0; i < cellint; i++) {
		L.insert(i, i) = nbdevoisins(i)+1;
		Lbis(i, i) = nbdevoisins(i) + 1;
		cout << "Nb de voisins de " << i << " (position = "<< SpaceRep.Cellint[i].x<<" "<< SpaceRep.Cellint[i].y<<" "<< SpaceRep.Cellint[i].z<<") : " << nbdevoisins(i) << endl;
	}		

	//cout << Lbis << endl;
	// Set the matrix B Aeq and Q
	Eigen::SparseMatrix<double> Zspace;
	double taille_x = (SpaceRep.Xmax - SpaceRep.Xmin) / (MAX - 1);
	double taille_y = (SpaceRep.Ymax - SpaceRep.Ymin) / (MAX - 1);
	double taille_z = (SpaceRep.Zmax - SpaceRep.Zmin) / (MAX - 1);
	double taille_voxel = taille_x * taille_y * taille_z;

	B.resize(cellint);
	Zspace.resize(cellint, 1);
	Aeq.resize(4,cellint);
	for (int i = 0;i < cellint;i++) {
		double xspace = SpaceRep.Xmin + SpaceRep.Cellint[i].x * taille_x;
		double yspace = SpaceRep.Ymin + SpaceRep.Cellint[i].y * taille_y;
		double zspace = SpaceRep.Zmin + SpaceRep.Cellint[i].z * taille_z;
		B(i) = gammaI * (2 * zspace * zspace - xspace * xspace - yspace * yspace);
		//cout<<"Point B["<<i<<"]= "<<B(i)<<endl;
		Zspace.insert(i, 0) = zspace;

		Aeq.insert(0, i) = xspace;
		Aeq.insert(1, i) = yspace;
		Aeq.insert(2, i) = xspace * zspace;
		Aeq.insert(3, i) = yspace * zspace;
	}
	Q.resize(cellint, cellint);
	Q = 2*gammaC * Zspace * Zspace.transpose();


	//Beq : all equals 0
	Beq.resize(4);
	Beq(0) = Beq(1) = Beq(2) = Beq(3) = 0;


	// Lower and upper bound
	lx = VectorXd::Zero(cellint,1);
	ux = VectorXd::Ones(cellint,1);

	// Set Z just to save time in execution

	Z = VectorXd::Ones(cellint);
	Z = Z * .5;

	// Nothing for Aieq, Bieq, b and bc
	igl::active_set_params as;
	int max_iter = 10;
	//as.Auu_pd = false;
	//as.inactive_threshold = 0.0000001;
	//as.constraint_threshold = 0.0000001;
	//as.solution_diff_threshold = 0.0000001;
	double gammaL = 1;


	cout << "Lancement du solveur" << endl;
	Q = Q + gammaL * L;

	for (int iteration = 1;iteration <= max_iter;iteration++) {
		as.max_iter = iteration;
		Z = VectorXd::Ones(cellint);
		Z = Z * .5;
 		igl::active_set(Q, B, b, bc, Aeq, Beq, Aieq, Bieq, lx, ux, as, Z);
		double erreur = 0;
		for (int i = 0; i < cellint; i++) {
			if (Z(i) < lx(i, 0)) {
				//cout << "Indice i = " << i << " avec Z(i) = " << Z(i) << endl;
				erreur += lx(i, 0) - Z(i);
			}

			if (Z(i) > ux(i, 0)) {
				erreur += -ux(i, 0) + Z(i);
				//cout << "Indice i = " << i << " avec Z(i) = " << Z(i) << endl;
			}

		}
		cout << "Itération numéro " << iteration << " faite : erreur cumulee : " << erreur << endl;
		if (erreur == 0) break;
	}


	//for (int i = 0;i < cellint;i++) {
		//cout << "i = " << i << " Z(i)=" << Z(i) << " lx(i,0) = " << lx(i, 0) << " ux(i,0)= " << ux(i, 0) << endl;
	//}
	cout << endl;



	// OLD Code starts here
	HalfedgeBuilder* builder = new HalfedgeBuilder();  //
	HalfedgeDS he = builder->createMesh(V.rows(), F);  //
	// Compute normals
	igl::per_face_normals(V,F,N_faces);
	igl::per_vertex_normals(V,F,N_vertices);
	lib_vertexNormals();
	vertexNormals(he); 

	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.data().set_mesh(V, F);  //
	viewer.data().set_normals(N_faces);  //
	std::cout<<
    "Press '1' for per-face normals calling pre-defined functions of LibiGL."<<std::endl<<
	"Press '2' for per-vertex normals calling pre-defined functions of LibiGL."<<std::endl<<
    "Press '3' for lib_per-vertex normals using face-vertex structure of LibiGL ."<<std::endl<<
	"Press '4' for HE_per-vertex normals using HalfEdge structure."<<std::endl;

	//  Z colors
	VectorXd Z2= VectorXd(V.rows(),1);
	Z2 = V.col(2);

	// Assign per-vertex colors
	MatrixXd C;
	igl::jet(Z2,true,C);
	viewer.data().set_colors(C);  // Add per-vertex colors

	viewer.launch(); // run the viewer
}
