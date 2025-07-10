from mpi4py import MPI
import numpy as np
import cv2
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Processo 0 lê a imagem
if rank == 0:
    img = cv2.imread("maspcomruido.jpg", 0)
    altura, largura = img.shape
else:
    img = None
    altura = largura = 0

# Broadcast das dimensões
altura = comm.bcast(altura, root=0)
largura = comm.bcast(largura, root=0)

# 1) calcula quantas linhas cada processo receberá (com sobra distribuída)
linhas_base = altura // size
extras      = altura % size
linhas = [linhas_base + (1 if i < extras else 0) for i in range(size)]

# 2) converte isso para contagem de elementos e deslocamentos em elementos
sendcounts = [l * largura for l in linhas]
displs      = [sum(sendcounts[:i]) for i in range(size)]

# 3) prepara buffer local
recvbuf = np.zeros(sendcounts[rank], dtype='uint8')
flat = img.flatten() if rank == 0 else None

# 4) dispara o Scatterv
comm.Scatterv([flat, sendcounts, displs, MPI.UNSIGNED_CHAR],
              recvbuf, root=0)

# 5) reconstrói a sub-imagem 2D
sub_img = recvbuf.reshape((linhas[rank], largura))

comm.Barrier()
t0 = MPI.Wtime()
filtered = cv2.blur(sub_img, (3,3))
t1 = MPI.Wtime()

# 6) achata de volta e recolhe com Gatherv usando mesmos sendcounts/displs
flat_f = filtered.flatten()
if rank == 0:
    result = np.empty(altura * largura, dtype='uint8')
else:
    result = None

comm.Gatherv(flat_f,
             [result, sendcounts, displs, MPI.UNSIGNED_CHAR],
             root=0)

# 7) no root, reconstroi, mede média global e faz seq. para speedup
if rank == 0:
    img_par = result.reshape((altura, largura))
    soma_p = np.sum(img_par)
    media_p = soma_p / (altura * largura)

    t0s = time.time()
    img_seq = cv2.blur(img, (3,3))
    t1s = time.time()
    speedup = (t1s - t0s) / (t1 - t0)

    print(f"Média paralela: {media_p:.2f}")
    print(f"Tempo par: {t1 - t0:.4f}s  seq: {t1s - t0s:.4f}s  speedup: {speedup:.2f}")

    cv2.imshow("Resultado MPI", img_par)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
