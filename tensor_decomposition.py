
import numpy as np


def phi_tensor(s,r,u,t):
    ''' phi tensor, defined above.
    it's the product of the tensor s with the matrices r,u,t
    has the same dimensions of A'''
    return np.einsum('ijk,ai,bj,ck->abc',s,r,u,t)
def phi_ijk(s,r,u,t,a,b,c):
    '''(i,j,k)th component of phi,
    i.e. the contraction of s with r,u,t tensors'''
    return np.einsum('ijk,i,j,k->',s,r[a,:],u[b,:],t[c,:])
def phi_ijk_2(s,ri,uj,tk):
    '''a possibly more optimized version
    ri,uj,tk=r[i,:],u[j,:],t[k,:]'''
    return np.einsum('ijk,i,j,k->',s,ri,uj,tk)
def rut(r,u,t,a,b,c):
    '''tensor product of i,j,k th components
    of r,u,t'''
    return np.einsum('i,j,k->ijk',r[a,:],u[b,:],t[c,:])
def compute_loss(a,x,y,s,r,u,t,f,g,lamb1=1,lamb2=1,lamb3=1):
    '''implementation of the loss function itself.
    the loss is composed by the sum of 4 different terms,
    they will be computed separately
    1st term is the one involving a
    2nd involves x
    3rd involves y
    4 is the regularization
    '''
    #problem: should sum only for i,j,k for which a_ijk >0: so I should implements a mask
    phi=phi_tensor(s,r,u,t)
    mask=np.where(a > 0, 1, 0)
    first=np.linalg.norm((a-phi)*mask)**2
    second=np.linalg.norm(x-t@g)**2
    third=np.linalg.norm(y-r@f)**2
    fourth=np.linalg.norm(s)**2+ np.linalg.norm(r)**2+ np.linalg.norm(u)**2+ np.linalg.norm(t)**2+ np.linalg.norm(f)**2+ np.linalg.norm(g)**2
    loss=0.5*first+0.5*lamb1*second+lamb2*0.5*third+lamb3*0.5*fourth
    return loss

def main():
    return 0

if __name__=="__main__":
    main()