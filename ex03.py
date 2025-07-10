from mpi4py import MPI
import numpy as np
import cv2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    image = cv2.imread("maspcomruido.jpg", 0)
    altura, largura = image.shape
else:
    image = None
    altura = largura = 0

altura = comm.bcast(altura, root=0)
largura = comm.bcast(largura, root=0)

lines_per_proc = [altura // size + (1 if i < altura % size else 0) for i in range(size)]
displs = [sum(lines_per_proc[:i]) for i in range(size)]
sendcounts = [l * largura for l in lines_per_proc]

recv_buffer = np.zeros(sendcounts[rank], dtype='uint8')
if rank == 0:
    flat_image = image.flatten()
else:
    flat_image = None

comm.Scatterv([flat_image, sendcounts, displs, MPI.UNSIGNED_CHAR], recv_buffer, root=0)
local_image = recv_buffer.reshape((lines_per_proc[rank], largura))

# Cada processo calcula a soma local dos pixels
soma_local = np.sum(local_image, dtype=np.uint64)
qtd_local = local_image.size

# Redução para obter soma total e total de pixels
soma_total = comm.reduce(soma_local, op=MPI.SUM, root=0)
qtd_total = comm.reduce(qtd_local, op=MPI.SUM, root=0)

if rank == 0:
    media = soma_total / qtd_total
    print("Média da intensidade dos pixels:", media)

    if media >= 200:
        print("Imagem clara/brilhante.")
    elif media >= 100:
        print("Imagem com tom médio de cinza.")
    else:
        print("Imagem escura.")