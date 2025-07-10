from mpi4py import MPI
import numpy as np
import cv2

# Obtém o comunicador global, rank e número total de processos
comm = MPI.COMM_WORLD    # COMM_WORLD engloba todos os processos lançados
rank = comm.Get_rank()   # obtém o identificador (rank) do processo atual
size = comm.Get_size()   # obtém o número total de processos MPI

# No processo 0, carrega a imagem e obtém dimensões; nos demais, inicializa variáveis
if rank == 0:
    image = cv2.imread("maspcomruido.jpg", 0) # lê imagem em tons de cinza
    altura, largura = image.shape # obtém número de linhas (altura) e colunas (largura)
else:
    image = None # processos não-zero não carregam a imagem
    altura = largura = 0 # inicializa dimensões como 0

altura = comm.bcast(altura, root=0) # envia altura do processo 0 a todos
largura = comm.bcast(largura, root=0) # envia largura do processo 0 a todos

lines_per_proc = [altura // size + (1 if i < altura % size else 0) for i in range(size)] # Divide a altura em partes quase iguais entre processos
# Calcula deslocamentos (despls) e quantidades de elementos (sendcounts)
displs = [sum(lines_per_proc[:i]) for i in range(size)] # linhas acumuladas antes de cada processo
sendcounts = [l * largura for l in lines_per_proc] # total de pixels (elementos) para cada processo

# Prepara buffer de recepção achatado para cada processo
recv_buffer = np.zeros(sendcounts[rank], dtype='uint8') # array 1D de zeros do tamanho esperado
if rank == 0:
    flat_image = image.flatten() # converte matriz 2D da imagem em vetor 1D
else:
    flat_image = None # demais processos não precisam ter dados iniciais

comm.Scatterv([flat_image, sendcounts, displs, MPI.UNSIGNED_CHAR], recv_buffer, root=0) # Scatterv reparticiona partes do vetor achatado para recv_buffer de cada processo
local_image = recv_buffer.reshape((lines_per_proc[rank], largura)) # Reconstrói a subimagem 2D a partir do vetor recebido

# Cada processo calcula a soma local dos pixels
soma_local = np.sum(local_image, dtype=np.uint64) # soma das intensidades da subimagem
qtd_local = local_image.size # número de pixels na subimagem

soma_total = comm.reduce(soma_local, op=MPI.SUM, root=0) # Redução para somar todas as somas locais em soma_total no processo 0
qtd_total = comm.reduce(qtd_local, op=MPI.SUM, root=0) # Redução para somar todas as quantidades locais em qtd_total no processo 0

# No processo 0, calcula a média global e classifica o brilho
if rank == 0:
    media = soma_total / qtd_total # média de intensidade de todos os pixels
    print("Média da intensidade dos pixels:", media)

    # Classificação simples por faixas de brilho
    if media >= 200:
        print("Imagem clara/brilhante.")
    elif media >= 100:
        print("Imagem com tom médio de cinza.")
    else:
        print("Imagem escura.")
