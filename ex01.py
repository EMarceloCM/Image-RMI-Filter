from mpi4py import MPI
import numpy as np
import cv2
import time

comm = MPI.COMM_WORLD # comunicador MPI que engloba todos os processos
rank = comm.Get_rank() # identificador (0 a size-1) do processo atual
size = comm.Get_size() # número total de processos MPI

# No processo 0, carrega a imagem e obtém dimensões; nos demais, inicializa variáveis
if rank == 0:
    img = cv2.imread("maspcomruido.jpg", 0) # lê imagem em tons de cinza
    altura, largura = img.shape # obtém altura e largura da imagem
else:
    img = None # processos não-zero não carregam a imagem
    altura = largura = 0 # inicializa dimensões como 0

# Broadcast das dimensões para todos os processos\ altura = comm.bcast(altura, root=0)
altura = comm.bcast(altura, root=0) # envia altura do processo 0 a todos
largura = comm.bcast(largura, root=0) # envia largura do processo 0 a todos

# Divide a altura em partes quase iguais entre processos
alturas_divididas = [altura // size + (1 if i < altura % size else 0) for i in range(size)]
deslocamentos = [sum(alturas_divididas[:i]) * largura for i in range(size)] # Calcula deslocamentos (offset) em pixels para o Scatterv/Gatherv
quantidade_pixels = [h * largura for h in alturas_divididas] # quantidade de pixels que cada processo vai receber/mandar

# Prepara buffer local para receber parte achatada da imagem
sub_img_flat = np.zeros(quantidade_pixels[rank], dtype='uint8')
if rank == 0:
    img_flat = img.flatten() # achata toda a imagem no processo 0
else:
    img_flat = None # demais processos não têm dados iniciais

# Distribui partes da imagem achatada para cada processo
comm.Scatterv([img_flat, quantidade_pixels, deslocamentos, MPI.UNSIGNED_CHAR],
              sub_img_flat, root=0) # Scatterv permite tamanhos variáveis por processo
sub_img = sub_img_flat.reshape((alturas_divididas[rank], largura)) # Reconstrói a sub-imagem 2D a partir do vetor achatado

comm.Barrier() # Sincroniza todos os processos antes de começar a contar tempo
# Mede tempo paralelo usando MPI.Wtime
t0_par = MPI.Wtime() # instante inicial (alta precisão MPI)
new_sub_img = cv2.blur(sub_img, (3, 3)) # aplica filtro média 3x3 na sub-imagem
t1_par = MPI.Wtime() # instante final
tempo_par = t1_par - t0_par # tempo gasto na filtragem paralela

# Achata a sub-imagem filtrada para enviar de volta
new_sub_img_flat = new_sub_img.flatten()
if rank == 0:
    result_flat = np.empty(altura * largura, dtype='uint8') # buffer para reconstruir imagem inteira
else:
    result_flat = None

# Coleta partes filtradas de volta no processo 0
comm.Gatherv(new_sub_img_flat,
             [result_flat, quantidade_pixels, deslocamentos, MPI.UNSIGNED_CHAR],
             root=0)

# Calcula soma local dos pixels filtrados para média global
media_local = np.sum(new_sub_img)
# Reduz (soma) todas as somas locais no processo 0
soma_global = comm.reduce(media_local, op=MPI.SUM, root=0)
# Reduz (soma) o número total de pixels para média
quantidade_total_pixels = comm.reduce(new_sub_img.size, op=MPI.SUM, root=0)

if rank == 0:
    media_final = soma_global / quantidade_total_pixels # média global dos pixels

    # --- Seção sequencial para comparação de tempo ---
    seq_img = None
    t0_seq = time.time() # tempo inicial sequencial
    seq_img = cv2.blur(img, (3, 3)) # aplica filtro média 3x3 sequencialmente na imagem inteira
    t1_seq = time.time() # tempo final sequencial
    tempo_seq = t1_seq - t0_seq # tempo gasto na filtragem sequencial

    # Exibe resultados de média e tempos
    print(f"Média da imagem (paralela): {media_final:.2f}")
    print(f"Tempo paralelo: {tempo_par:.4f} s")
    print(f"Tempo sequencial: {tempo_seq:.4f} s")
    print(f"Speedup (seq/par): {tempo_seq/tempo_par:.2f}")

    # Reconstrói a imagem filtrada e exibe
    resultado = result_flat.reshape((altura, largura))
    cv2.imshow("Imagem Filtrada MPI", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
