#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <ostream>
#include <igl/readOFF.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <math.h>
#include <igl/octree.h>

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
#define MAX 15

struct Point {
	int x;
	int y;
	int z;
};


MatrixXd V;
MatrixXi F;


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

			   				 

// ------------ main program ----------------
int main(int argc, char *argv[]) {

	igl::readOFF("../data/toupie_3d.off",V,F);
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
	V = V * 10;

	XYZminmax = compute_min_max_on_axes(V);
	Xmin = XYZminmax(0, 0);	Xmax = XYZminmax(0, 1); Ymin = XYZminmax(1, 0); Ymax = XYZminmax(1, 1); Zmin = XYZminmax(2, 0); Zmax = XYZminmax(2, 1);
	cout << "Après rotation et translation : " << endl;
	cout << "L'axe x va de " << Xmin << " a " << Xmax << endl;
	cout << "L'axe y va de " << Ymin << " a " << Ymax << endl;
	cout << "L'axe z va de " << Zmin << " a " << Zmax << endl;

	std::vector<std::vector<int > > O_PI;
	Eigen::MatrixXi O_CH;
	Eigen::MatrixXd O_CN;
	Eigen::VectorXd O_W;
	igl::octree(V, O_PI, O_CH, O_CN, O_W);

	int Profondeur[10500];
	int Objparfeuille[10500];
	for (int i = 0;i < 10500;i++) {
		Profondeur[i] = 0;		   Objparfeuille[i] = 0;
	}
	cout << "O_CH :" << O_CH.rows()<<"  "<<O_CH.cols() << endl;
	cout << "O_CN :" << O_CN.rows() << "  " << O_CN.cols() << endl;
	cout << "O_W :" << O_W.rows() << "  " << O_W.cols() << endl;

	for (int i = 0;i < O_W.rows();i++) {
		if (O_CH(i, 0) == -1) {	// C'est une feuille
			int prof = 48 / O_W[i];
			Profondeur[prof]++;
			int nbobj = O_PI[i].size();
			Objparfeuille[nbobj]++;

		}
	}
	for (int i= 0;i < 1025;i++)
		if (Profondeur[i] > 0) cout << "Profondeur " << i << " : " << Profondeur[i] << endl;


	for (int i = 0;i < 1025;i++)
		if (Profondeur[i] > 0) cout << "Octree ayant " << i << " points : " << Objparfeuille[i] << endl;


	igl::opengl::glfw::Viewer viewer; // create the 3d viewer
	viewer.data().set_mesh(V, F);  //
	viewer.data().add_points(O_CN, Eigen::RowVector3d(1, 0, 0));
	viewer.launch(); // run the viewer
}
