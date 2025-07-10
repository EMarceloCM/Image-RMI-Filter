from mpi4py import MPI
import numpy as np
import cv2
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    img = cv2.imread("maspcomruido.jpg", 0)
    altura, largura = img.shape
else:
    img = None
    altura = largura = 0

altura = comm.bcast(altura, root=0)
largura = comm.bcast(largura, root=0)

alturas_divididas = [altura // size + (1 if i < altura % size else 0) for i in range(size)]
deslocamentos = [sum(alturas_divididas[:i]) * largura for i in range(size)]
quantidade_pixels = [h * largura for h in alturas_divididas]

sub_img_flat = np.zeros(quantidade_pixels[rank], dtype='uint8')
if rank == 0:
    img_flat = img.flatten()
else:
    img_flat = None

comm.Scatterv([img_flat, quantidade_pixels, deslocamentos, MPI.UNSIGNED_CHAR],
              sub_img_flat, root=0)
sub_img = sub_img_flat.reshape((alturas_divididas[rank], largura))

comm.Barrier()
t0_par = MPI.Wtime()
new_sub_img = cv2.blur(sub_img, (3, 3))
t1_par = MPI.Wtime()
tempo_par = t1_par - t0_par

new_sub_img_flat = new_sub_img.flatten()
if rank == 0:
    result_flat = np.empty(altura * largura, dtype='uint8')
else:
    result_flat = None

comm.Gatherv(new_sub_img_flat,
             [result_flat, quantidade_pixels, deslocamentos, MPI.UNSIGNED_CHAR],
             root=0)

media_local = np.sum(new_sub_img)
soma_global = comm.reduce(media_local, op=MPI.SUM, root=0)
quantidade_total_pixels = comm.reduce(new_sub_img.size, op=MPI.SUM, root=0)

if rank == 0:
    media_final = soma_global / quantidade_total_pixels

    seq_img = None
    t0_seq = time.time()
    seq_img = cv2.blur(img, (3, 3))
    t1_seq = time.time()
    tempo_seq = t1_seq - t0_seq

    print(f"Média da imagem (paralela): {media_final:.2f}")
    print(f"Tempo paralelo: {tempo_par:.4f} s")
    print(f"Tempo sequencial: {tempo_seq:.4f} s")
    print(f"Speedup (seq/par): {tempo_seq/tempo_par:.2f}")

    resultado = result_flat.reshape((altura, largura))
    cv2.imshow("Imagem Filtrada MPI", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()