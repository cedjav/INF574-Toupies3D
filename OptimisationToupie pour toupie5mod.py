import numpy as np
from numpy import linalg
import os
import math

#Chargement du .OFF
def readfile(file):
    os.chdir('c:\\Users\cedri\OneDrive\Bureau\Formation\INF574\Projet\TOUPIE5MODIFIEE')
    f=open(file)
    line=f.readline()
    line=f.readline()
    line=line.split()
    NumberofV=int(line[0]) #Number of vertices
    NumerofF=int(line[1])

    V=[]
    for i in range(NumberofV):
        line=f.readline()
        line=line.split()
        V.append(float(line[0]))
        V.append(float(line[1]))
        V.append(float(line[2]))

    F=[]
    for i in range(NumerofF):
        line=f.readline()
        line=line.split()
        F.append(int(line[1]))
        F.append(int(line[2]))
        F.append(int(line[3]))
        
    V=np.asarray(V)
    V=V.reshape(-1,3)
    F=np.array(F)
    F=F.reshape(-1,3)

    f.close
    return V,F


#Calcul du vecteur s
def barre(a):
    b=np.zeros(3)
    b[0]=a[1]
    b[1]=a[2]
    b[2]=a[0]
    return b
def computeSonsurface(V, F):
    s=np.zeros(10)
    for i in range(F.shape[0]):
        a = V[F[i,0],:]
        b = V[F[i,1],:]
        c = V[F[i,2],:]
        u = b - a
        v = c - a
        u = -u
        n = -np.cross(u,v)
        h1 = a + b + c
        h2 = a*a + b * (a + b)
        h3 = h2 + (c*h1)
        h4 = a * a * a + (b * h2) + (c * h3)
        h5 = h3 + (a * (h1 + a))
        h6 = h3 + (b * (h1 + b))
        h7 = h3 + (c * (h1 + c))
        h8 = (barre(a) * h5) + (barre(b) * h6) + (barre(c) * h7)
        s[0] += (n*h1)[0]
        xx = n * h3
        s[1] += xx[0]
        s[2] += xx[1]
        s[3] += xx[2]
        xx = n*h8
        s[4] += xx[0]
        s[5] += xx[1]
        s[6] += xx[2]
        xx = n*h4
        s[7] += xx[0]
        s[8] += xx[1]
        s[9] += xx[2]

    s[0] /= 6
    s[1] /= 24
    s[2] /= 24
    s[3] /= 24
    s[4] /= 120
    s[5] /= 120
    s[6] /= 120
    s[7] /= 60
    s[8] /= 60
    s[9] /= 60
    if (s[0] < 0): s=-s
    return s

# Rotation Transalation et symmétrie pour poser la toupie correctement
def repositionnetoupie(V):
    V1=np.zeros(V.shape)
    V1[:,0]=V[:,1]-14
    V1[:,1]=V[:,2]-14
    V1[:,2]=48-V[:,0]
    #print("Pour la toupie :")
    #print("Nouvelles coordonnées sur x : de ",np.amin(V1[:,0])," à ",np.amax(V1[:,0]))
    #print("Nouvelles coordonnées sur y : de ",np.amin(V1[:,1])," à ",np.amax(V1[:,1]))
    #print("Nouvelles coordonnées sur z : de ",np.amin(V1[:,2])," à ",np.amax(V1[:,2]))
    #print("")
    return V1


# Calcul d'un objet "Cube modifé" selon 6 paramètres ajustables
def repositionnecube(Vcube,bas,haut,tetamin,tetamax,longbas,longhaut):
    
    V2=[[bas,14,14],
        [bas,14+math.cos(tetamin)*longbas,14+math.sin(tetamin)*longbas],
        [bas,14+math.cos((tetamin+tetamax)/2)*longbas,14+math.sin((tetamin+tetamax)/2)*longbas],
        [bas,14+math.cos(tetamax)*longbas,14+math.sin(tetamax)*longbas],
        [haut,14,14],
        [haut,14+math.cos(tetamin)*longhaut,14+math.sin(tetamin)*longhaut],
        [haut,14+math.cos((tetamin+tetamax)/2)*longhaut,14+math.sin((tetamin+tetamax)/2)*longhaut],
        [haut,14+math.cos(tetamax)*longhaut,14+math.sin(tetamax)*longhaut]]
           
    V2=np.asarray(V2)
    V2=V2.reshape(-1,3)
    V1=np.zeros(Vcube.shape)
    
    V1[:,0]=V2[:,1]-14
    V1[:,1]=V2[:,2]-14
    V1[:,2]=48-V2[:,0]

    #print("Pour le cube :")
    #print("Nouvelles coordonnées sur x : de ",np.amin(V1[:,0])," à ",np.amax(V1[:,0]))
    #print("Nouvelles coordonnées sur y : de ",np.amin(V1[:,1])," à ",np.amax(V1[:,1]))
    #print("Nouvelles coordonnées sur z : de ",np.amin(V1[:,2])," à ",np.amax(V1[:,2]))
    #print("")
    return V1

#Calcul de la fonction d'erreur / bool=VRAI Pour afficher
def getinfospin(s,bool):
    Masse=s[0]
    if bool:
        print("La masse est de : ",Masse)
    Center=s[1:4]/Masse
    if bool:
        print("Position du centre de masse :",Center)
    Inertia=np.zeros((3,3))
    Inertia[0,0]=s[8]+s[9]
    Inertia[0,1]=Inertia[1,0]=-s[4]
    Inertia[1,1]=s[7]+s[9]
    Inertia[0,2]=Inertia[2,0]=-s[6]
    Inertia[1,2]=Inertia[2,1]=-s[5]
    Inertia[2,2]=s[7]+s[8]
    if bool:
        print("Condition n°1 Sx=0 et là on a : ",s[1])
        print("Condition n°2 Sy=0 et là on a : ",s[2])
        print("Condition n°3 Sxz=0 et là on a : ",s[6])
        print("Condition n°4 Syz=0 et là on a : ",s[5])
    
    A=Center.reshape(3,1)*Center.reshape(1,3)
    b=(Center.reshape(1,3)@Center.reshape(3,1))[0,0]
    Identite=np.array([[1,0,0],[0,1,0],[0,0,1]])
    Inertiacentre=Inertia+Masse*(A-b*Identite)
    W,Rot=linalg.eig(Inertiacentre[0:2,0:2])
    coss=Rot[0,0]
    sinn=Rot[1,0]
    condition5=coss*sinn*(s[7]-s[8])+(coss**2-sinn**2)*s[5]
    if bool : 
        print("Condition n°5 : la valeur suivante doit être nulle : ",condition5)
    Ia=W[0]
    Ib=W[1]
    Ic=Inertia[2,2]
    Fyoyo=((Ia/Ic)**2+(Ib/Ic)**2)
    Ftop=(Center[2]*Masse)**2
    noteglobale=s[1]**2+s[2]**2+abs(s[6])+abs(s[5])+abs(condition5)
    
    if bool:
        print("Note globale (0=parfait)",noteglobale)
    return Fyoyo,Ftop,noteglobale

# Calcul de la surface de (V,F) - Pas utilisé ici
def surface(V,F):
    surf=0
    for i in range(F.shape[0]):
        a = V[F[i,0],:]
        b = V[F[i,1],:]
        c = V[F[i,2],:]
        u = b - a
        v = c - a
        n = np.cross(u,v)
        surf=surf+np.linalg.norm(n)
    return surf/2
        

# PROGRAMME PRINCIPAL
Vtoupie,Ftoupie=readfile("toupie_mod5.off")
Vtoupie=repositionnetoupie(Vtoupie)
stoupie=computeSonsurface(Vtoupie,Ftoupie)
print("**********TOUPIE 5 PLEINE***********")
getinfospin(stoupie,True)   #Calcule les paramètres et l'erreur sur la toupie modifiée pleine (théorique)

Vtoupie_int,Ftoupie_int=readfile("interieur_toupie_mod5_epaisseur15.off")
Vtoupie_int=repositionnetoupie(Vtoupie_int)
stoupie_int=computeSonsurface(Vtoupie_int,Ftoupie_int)
stotal=stoupie-stoupie_int*(1-.33)
print("**********TOUPIE 5 MODIFIEE ET REMPLIE SEULEMENT A 33%***********")
getinfospin(stotal,True)  #Calcule les paramètres et l'erreur sur la toupie modifiée (réelle) 

Vcube,Fcube=readfile("cube_tri_modif3.off") # Charge l'objet de départ (on ne conserve que sa topologie)
best_note=1e12

print("")
print("Début de l'optimisation")
note=[]
MAX=10
for i in range(MAX):
    for j in range(MAX):
        for k in range (MAX):
            for l in range (MAX):
                for m in range(MAX):
                    for n in range (MAX):
                        haut=35-i*2/MAX
                        bas=19+j*2/MAX
                        teta1=3+.5*k/MAX
                        teta2=4.5-0.5*l/MAX
                        l1=12-m*10/MAX
                        l2=11-n*10/MAX
                        
                        NVcube=repositionnecube(Vcube,haut,bas,teta1,teta2,l1,l2)
                        scube=computeSonsurface(NVcube,Fcube)
                        a,b,nn=getinfospin(stotal-.33*(scube),False)
                        note.append(nn)
                        if nn<best_note:
                            best_i=i
                            best_j=j
                            best_k=k
                            best_l=l
                            best_m=m
                            best_n=n
                            best_note=nn
                            print("On a mieux avec",best_i,best_j,best_k,best_l,best_m,best_n,"   - nouvelle meilleure note : ",best_note)
        
print("Meilleur résultat avec : ")

haut=35-best_i*2/MAX
bas=19+best_j*2/MAX
teta1=3+.5*best_k/MAX
teta2=4.5-0.5*best_l/MAX
l1=12-best_m*10/MAX
l2=11-best_n*10/MAX
                        
print("haut=",haut)
print("bas=",bas)
print("teta1=",teta1)
print("teta2=",teta2)
print("l1=",l1)
print("l2=",l2)
print("Note : ",best_note)
NVcube_final=repositionnecube(Vcube,haut,bas,teta1,teta2,l1,l2)
scubefinal=computeSonsurface(NVcube_final,Fcube)

print("**********'CUBE' FINAL à 33% ***********")
getinfospin(scubefinal*.33,True)

print("**********TOUPIE 5 MODIFIEE ET REMPLIE SEULEMENT A 33% - 'CUBE' FINAL à 33%***********")
getinfospin(stotal-.33*scubefinal,True)

