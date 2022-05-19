from cmath import cos, sin, sqrt
from gettext import find
import random as r
from re import A
from typing import Any
import matplotlib.pyplot as pl
import math
import numpy as np


#methode pour trouver le minimum d'une liste

def findMin(list):
    x0=list[0]
    k=0
    for i in range (len(list)):
        if (x0>list[i]):
             x0=list[i]
             k=i
    return k

#trouver le maximun d'une liste
def findMax(list):
    x0=list[0]
    k=0
    for i in range (len(list)):
        if (x0<list[i]):
             x0=list[i]
             k=i
    return k


def f(x):
    return x**3-3*x**2+2*x+5

#pour trouver le maximum de f, on peut chercher le minimum  de l'inverse de f avec les methodes des balayages
#
def inv(x):
    return 1/f(x)
        
def list():
    list=[]
    for i in range (5):
        list.append(i)
    return list


#methode a balayage a pas constant
def balayageCstMin(f,a,b,n):
    ylist=[]
    xlist=[]
    delta=(b-a)/n
    x=a
    for i in range (n+1):
        x=x+i*delta
        y=f(x)
        xlist.append(x)
        ylist.append(y)
    #pl.plot(xlist,ylist)
    #pl.show()
    return (xlist[findMin(ylist)],ylist[findMin(ylist)])

def complexe_modulo(z):
    a = z.real
    b = z.imag
    return math.sqrt(a**2+b**2)

#methode par balayage aleatoire 
def balayageAleaMin(function,a,b,n):
    xlist=[]
    ylist=[]
    for i in range(n+1):
        x=r.uniform(a,b)
        xlist.append(x)
    xlist.sort() #on ordonne les abcisses dans l'ordre croissant pour avoir un graphe clair
    for i in range (n+1): #remplassage de la liste des ordonnes
        y=function(xlist[i])
        ylist.append(y)  
    #pl.plot(xlist,ylist)
    #pl.show()
    return (xlist[findMin(ylist)],ylist[findMin(ylist)]) #f(xlist[findMin(ylist)]) pour trouver le maimum de f cad le minimum de l'inverse de f

#print("le resultat obtenu par balayage constant ",balayageCstMin(f,0,3,100))
#print("le resultat obtenu par balayage aleatoire ",balayageAleaMin(f,0,3,20))



def maxBalayageCst(function,a,b,n):
    ylist=[]
    xlist=[]
    delta=(b-a)/n
    x=a
    for i in range (n+1):
        x=x+i*delta
        y=function(x)
        xlist.append(x)
        ylist.append(y)
    pl.plot(xlist,ylist)
    pl.show()
    print(findMax(ylist))
    return (xlist[findMax(ylist)],f(xlist[findMax(ylist)]))

#tracer les courbes d'erreur relative
def ShowError(methode,f,a,b,n):
    nlist=[]
    Errorlist=[]
    indiceMin,min=balayageAleaMin(f,a,b,n)
    for i in range(1,n+1):
        nlist.append(i)
        x,y=methode(f,a,b,i)
        Errorlist.append(math.fabs(y-min))
    
    return nlist,Errorlist
'''''
lstx,lsty=ShowError(balayageCstMin,f,0,3,100)
lstx2,lsty2=ShowError(balayageAleaMin,f,0,3,100)
pl.plot(lstx,lsty,'r')
pl.plot(lstx2,lsty2,'g')
pl.show()
'''
def derivee(f,x):
    h=1*10**-6
    return (f(x+h)-f(x))/h

#derivee premiere centree
def DerPremiere(f,x):
    h=10**-4
    d=(f(x+h)-f(x-h))/2*h
    return d

def DerSeconde(f,x):
    h=10**-6
    d=(f(x+h)-2*f(x)+f(x-h))/h**2
    return d

def phi(t,x):
    return f(x+t*DerPremiere(f,x))

#methode du gradiant 1D 
#pour trouver le minimum d'une fonction 
def gradiant1D(f,a,b,u,n):
    xn=r.uniform(a,b)
    for i in range (n+1):
        xn=xn+u*derivee(f,xn)
    return xn,f(xn)

#print(gradiant1D(f,0,3,-1*10**-4,10000))
#print(balayageAleaMin(f,0,3,1000))
#print(DerPremiere(f,0))
#print(gradiant1D(f,0,3,-10**6,100))
#print(derivee(f,0))

def gab(a,b,x,y):
    return (x**2/a)+(y**2/b)

def g(x,y):
    a=2
    b=2/7
    return (x**2)/a+(y**2)/b

def h(x,y):
    return cos(x)*sin(y)

#represenation 3D des graphes de g
def graphe3dG(a,b):
    x=np.linspace(-100,100,100)
    y=np.linspace(-100,100,100)
    X,Y=np.meshgrid(x,y)
    Z=gab(a,b,X,Y)
    fig = pl.figure()
    ax = pl.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='jet', edgecolor = 'none')   
    ax.set_title("Graphe de g", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 11)
    pl.show()

def graphe3dH():
    x=np.linspace(-10,10,1000)
    y=np.linspace(-10,10,1000)
    X,Y=np.meshgrid(x,y)
    h2 = np.vectorize(h)
    Z=h2(X,Y)
    fig = pl.figure()
    ax = pl.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='jet', edgecolor = 'none')   
    ax.set_title("Graphe de h", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 11)
    pl.show()


#graphe3dG(2,2/7)
#graphe3dH()

#fonctions qui permettent de tracer les lignes de niveau de h et g
def ligneNiveauH():
    x=np.linspace(-5,5,1000)
    y=np.linspace(-5,5,1000)
    X,Y=np.meshgrid(x,y)
    h2 = np.vectorize(h)
    Z=h2(X,Y)
    pl.contour(X,Y,Z,10)
    

def ligneNiveauG(a,b):
    x=np.linspace(-10,10,1000)
    y=np.linspace(-10,10,1000)
    X,Y=np.meshgrid(x,y)
    #g = np.vectorize(gab)
    Z=gab(a,b,X,Y)
    pl.contour(X,Y,Z,200)
    


#ligneNiveauG(2,2/7)
#graphe3dH()
'''''
#derivee par rapport a la premiere variable x
def dfx(function,x,y):
    h=1*10**-4
    return (function(x+h,y)-function(x,y)/h)

#derivee par rapport a la deuxieme variable y
def dfy(function,x,y):
    h=1*10**-4
    return (function(x,y+h)-function(x,y)/h)

print(dfy(g,1,1))
'''
###########################################
#calcul a la min des derivées partielle de h
def dhx(x,y):
    return -sin(x)*sin(y)
def dhy(x,y):
    return cos(x)*cos(y)

#derivee partiel de g pour a=2 et b=2/è
def dgx(a,b,x,y):
    return 2*x/a
def dgy(a,b,x,y):
    return 2*y/b

def dgx1(x,y):
    a=2
    b=2/7
    return 2*x/a
def dgy1(x,y):
    a=2
    b=2/7
    return 2*y/b

###########################################


def norme(x,y,z):
    return math.sqrt(x**2+y**2+z**2)



#print(norme(0,2,0).real)

def gradpc(eps,m,u,x0,y0,df1,df2):
    a,b=2,2/7 #parametre de la fonction gab
    min = 0 #minimum du fonction qu'on veut etudier, pour g c'est 0
    nlist=[]
    xlist=[]
    ylist=[]
    Errorlist=[] #liste qui contient les erreurs absolues
    x,y=x0,y0 #initialisation du point de depart
    n=0 #initialisation du compteur
    while np.linalg.norm((np.real(df1(x,y)),np.real(df2(x,y))))>eps or n<m: #l'algorithme s'arrete si la norme du gradiant est inf a eps 
       #ou si les nombres d'iterations depassent m
        nlist.append(n)
        xlist.append(x)
        ylist.append(y)
        x=x+u*np.real(df1(x,y))   
        #df1 est la derivee de f par rapport a x en (xn,yn)                        
        y=y+u*np.real(df2(x,y)) 
        #df2 est la derivee de f par rapport a y en (xn,yn)
        #Errorlist.append(math.fabs(h(x,y)-min))
        n=n+1
    #pl.plot(nlist,xlist,'r')
    pl.plot(xlist,ylist,'r')  #pour afficher l'evolution de (xn,yn) dans le plan
    pl.title(u)
    ligneNiveauH()
    pl.plot(xlist,ylist,'b')  #pour afficher l'evolution de (xn,yn) dans le plan
    pl.title(u)
    #print(len(nlist))
    #pl.plot(nlist,Errorlist)  #pour afficher la courbe d'erreur relatif pour g(1,20)
    #pl.plot(nlist,ylist,'g')
    pl.show()
    

#gradpc(1*10**-6,200,-0.001,0,0,dhx,dhy)
#gradpc(1*10**-5,200,-0.001,7,1.5,dgx,dgy)
'''''
listpas=[-0.99,-0.5,-0.1,-0.01,-0.001]
for pas in listpas:
    #print(pas)
    gradpc(1*10**-5,120,pas,7,1.5,dgx,dgy)
    '''

#chercher un minimum d'une fonction
def gradamin(eps,m,u,x0,y0,f,df1,df2):
    n=0
    listx =[]
    listy=[]
    xn,yn=x0,y0
    while(norme(df1(xn,yn),(df2(xn,yn)),0)>eps):
        print("norme est egale a ",norme((df1(xn,yn)),(df2(xn,yn)),0))
        k=0
        d1=df1(xn,yn) #mise a jour du gradiant
        d2=df2(xn,yn)
        F1=f(xn+k*u*d1,yn+k*u*d2)
        F2=f(xn+(k+1)*u*d1,yn+(k+1)*u*d2)
        while((F2)<(F1)):     #on augmente k tant que f2>f1
            listx.append(xn)
            listy.append(yn)
            xn=xn+u*d1
            yn=yn+u*d2 #on avance avec le gradiant constant
            print(xn,yn)
            print("norme est egale a ",norme((df1(xn,yn)),(df2(xn,yn)),0))
            k=k+1
            F1=f(xn+k*u*d1,yn+k*u*d2) 
            F2=f(xn+(k+1)*u*d1,yn+(k+1)*u*d2)
        n=n+1
            
    #return listx,listy
    return xn,yn

#print(gradamin(1*10**-4,200,-0.001,7,1.5,g,dgx1,dgy1))




################################"
# Partie D
def G(A,y,secondMembre):
    return 2*(np.dot(A,y)-secondMembre)

#resolution systemes lineaires par algorithme d'inversion de matrices
def InvMatrice(eps,m,A,x0,b):
    k=1
    yk=x0
    #vecteur initial
    if(np.linalg.norm(G(A,yk,b)) !=0):
        pk=((np.linalg.norm(G(A,yk,b)))**2)/(2*(np.dot(np.dot(np.matrix.transpose(G(A,yk,b)),A),G(A,yk,b))))
    else:
        pk=0
    #calcul du pas pk
    
    yk1=yk-np.dot(pk,G(A,yk,b))

    while np.linalg.norm(yk1-yk)>eps or k<m :  #np.linalg.norm(yk1-yk)>eps 
        #criteres d'arret
        yk=yk1
        #print(yk1)
        if(np.linalg.norm(G(A,yk,b)) !=0):
            pk=(np.linalg.norm(G(A,yk,b)))**2/(2*(np.dot(np.dot(np.matrix.transpose(G(A,yk,b)),A),G(A,yk,b))))
        else:
            pk=0
        yk1=yk-np.dot(pk,G(A,yk,b))
        k=k+1
    
    return yk1



B0=np.array([3, 2])
Matrice=np.array([[1,1],[1, 0]])

InvMatrice(1*10**-8,50,Matrice,(0,0),B0)

#print(np.dot(Matrice,b0))

def tridiag(n,a,b,c):
    mat=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                mat[i][j]=b
            if j==i+1:
                mat[i][j]=c
            if i==j+1:
                mat[i][j]=a
    return mat

#print(tridiag(50,1,-2,1))
#Am=[[2,-1,0,0,0],[-1,2,-1,0,0],[0,-1,2,-1,0],[0,0,-1,2,-1],[0,0,0,-1,2,]]

#mat=np.array([[2,-1,0,0,0],[-1,2,-1,0,0],[0,-1,2,-1,0],[0,0,-1,2,-1],[0,0,0,-1,2,]])
############################################################
a,b=500,350
n=4
deltax=1/(n+1)
secondmembre=np.zeros(n)
secondmembre[0]=500/(deltax**2)
secondmembre[-1]=350/(deltax**2)
#print(secondmembre)

Am=(1/deltax**2)*tridiag(n,-1,2,-1)

tlist=np.linspace(0,1,n+2)

#InvMatrice(1*10-6,50,Am,np.zeros(50),secondmembre)

Tlist=np.zeros(n+2)
Tlist[0]=a
Tlist[-1]=b
Vinitial= np.zeros(n)
Vinitial[0]=a
Vinitial[-1]=b
Tlist[1:-1]=InvMatrice(1*10-6,n,Am,Vinitial,secondmembre)
#print(Tlist)
pl.plot(tlist,Tlist)
pl.xlabel('x')
pl.ylabel("T(x)")
pl.show()

###########################################################


def equationChaleur(x0,xf,n,c,f,a,b):
    #on construit une segmentation de x et t
    deltax=(xf-x0)/(n+1)
    xlist=np.linspace(x0,xf,n+2)
    #construction de la matrice tridiagonale
    mat= tridiag(n,-1,2-(c*deltax**2),-1)
    matrice=(1/deltax**2)*mat
    secmembre=np.full(n, f)
    secmembre[0]+=a/(deltax**2)
    secmembre[-1]+=b/(deltax**2)
    print(secmembre)
    Tlist=np.zeros(n+2)
    Tlist[0]=a
    Tlist[-1]=b
    Tlist[1:-1]=InvMatrice(1*10-6,n,matrice,np.zeros(n),secmembre)
    return xlist,Tlist


#pl.plot(equationChaleur(0,1,100,0,0,500,350))
#pl.show()

#print(tridiag(20,1,2,3))
    