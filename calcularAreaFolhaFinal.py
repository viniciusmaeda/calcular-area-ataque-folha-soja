# Programa para calcular a área atacada por lagartas em folhas de soja
# Autor: Vinícius de Aaújo Maeda


# importação das bibliotecas
import cv2 as cv
import sys
import numpy as np

print(cv.__version__)
print(np.__version__)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><>
def recortarRedimensionarImagem(img, escala):

  # primeiramente recorta a imagem para a área de interesse
  imgCortada = img[390:2890, 1190:3490]

  # define as dimensões com a nova escala
  novaAltura = int(imgCortada.shape[0] * escala)
  novaLargura = int(imgCortada.shape[1] * escala)

  # dimensão para a nova imagem
  dimensao = (novaLargura, novaAltura)

  # redimensiona o tamanho das imagens
  imagemNova = cv.resize(imgCortada, dimensao)

  # retora a imagem com a nova escala
  return imagemNova
# <><><><><><><><><><><><><><><><><><><><><><><><><><><>


# <><><><><><><><><><><><><><><><><><><><><><><><><><><>
# função para realizar o tratamento da imagem tais como:
# - transformar para escala de cinza
# - aplicar filtros de contraste e brilho
# - realizar a limiarização
# - aplicar a dilatação
def tratamentoImagem(img):

  # transformar para escala de cinza
  imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  # # altera o contraste e o brilho da imagem
  imgFiltroCB = cv.convertScaleAbs(imgGray, alpha=2, beta=50)

  # aplicação da limiarização separando o objeto do fundo
  ret, imgThresh = cv.threshold(imgFiltroCB, 150, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

  # aplicar dilatação
  kernel = np.ones((5, 5), np.uint8)
  imgDilatacao = cv.dilate(imgThresh, kernel, iterations = 1)

  # retorna a imagem tratada
  return imgDilatacao
# <><><><><><><><><><><><><><><><><><><><><><><><><><><>


# <><><><><><><><><><><><><><><><><><><><><><><><><><><>
def desenharContornoCalcularAreas(contorno, img):

  # razão entre a quantidade de pixel e a área de controle (barra vermelha)
  # obs. valor obtido através do Gimp, que pode ser obtido automaticamente
  RATIO_CM_PIXEL = 96 / 5 # 96 equivale à 5 cm
  RATIO_CM_PIXEL_QUADRADO = RATIO_CM_PIXEL ** 2

  # vetor para retornar com as áreas encontradas
  areasEncontradas = []

  # laço para percorrer todos os contornos encontrados
  for c in contorno:
    
    # obtem as coordenadas do retângulo
    (x, y, w, h) = cv.boundingRect(c)

    # descarta o objeto de controle e o retângulo da imagem
    if (x < 15):
      continue

    # obtém a área do contorno
    area = cv.contourArea(c)
    # calcula para cm2
    areaCm = round(area / RATIO_CM_PIXEL_QUADRADO, 2)

    # adiciona todas as áreas num array
    areasEncontradas.append(areaCm)

    # # para não mostrar as áreas atacadas
    # if (areaCm < 20):
    #   continue

    # desenha o contorno e mostra o valor da área
    img = cv.drawContours(img, [c], -1, (0, 255, 0), 1)
    # cv.putText(imgAntesRecortadaRedimensionada, 'Area: {} cm2'.format(str(areaCm)), (x+15, y+15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),1)

  areasEncontradas.sort()

  return areasEncontradas
# <><><><><><><><><><><><><><><><><><><><><><><><><><><>


# <><><><><><><><><><><><><><><><><><><><><><><><><><><>
if (__name__ == '__main__'):

  # carrega as imagens (antes e depois do ataque)
  imgAntes = cv.imread('./assets/20ACT.tif')
  imgDepois = cv.imread('./assets/20DCT.tif')

  # tratamento de erro ao abrir a imagem
  if imgAntes is None: sys.exit('Imagem antes do ataque não encontrada.')
  if imgDepois is None: sys.exit('Imagem antes do ataque não encontrada.')

  # redimensiona o tamanho das imagens
  imgAntesRecortadaRedimensionada = recortarRedimensionarImagem(imgAntes, 0.15)
  imgDepoisRecortadaRedimensionada = recortarRedimensionarImagem(imgDepois, 0.15)

  # realiza o tratamento das imagens
  imgAntesTratamento = tratamentoImagem(imgAntesRecortadaRedimensionada)
  imgDepoisTratamento = tratamentoImagem(imgDepoisRecortadaRedimensionada)

  # Encontrar os contornos presentes nas folhas (com e sem o ataque)
  contornosAntes, hierarchy = cv.findContours(imgAntesTratamento, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
  contornosDepois, hierarchyDepois = cv.findContours(imgDepoisTratamento, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

  # DEBUG: qntde de contornos encontradas nas imagens
  # print('Qtde de contornos (Antes): {}'.format(len(contornosAntes)))
  # print('Qtde de contornos (Depois): {}'.format(len(contornosDepois)))

  # desenhar o contorno e calcular as áreas encontradas
  areasAntes = desenharContornoCalcularAreas(contornosAntes, imgAntesRecortadaRedimensionada)
  areasDepois = desenharContornoCalcularAreas(contornosDepois, imgDepoisRecortadaRedimensionada)

  # DEBUG: mostra os valores dos arrays
  # print('array antes: ', areasAntes)
  # print('array depois: ', areasDepois)

  # calcula as áreas totais para ambas folhas (antes e depois do ataque)
  areaTotalAntesAtaque = areasAntes[-1]
  areaTotalDepoisAtaque = round(areaTotalAntesAtaque - (sum(areasDepois) - areasDepois[-1]), 2)

  # calcula a diferença em cm
  diferencaCentimetrosDepoisAtaque = round(areaTotalAntesAtaque - areaTotalDepoisAtaque, 2)
  # calcula a diferença percentual
  diferencaPercentualDepoisAtaque = round((diferencaCentimetrosDepoisAtaque * areaTotalAntesAtaque) / 100, 2)

  # DEBUG: mostra a diferença
  # print('diferenca em cm: ', diferencaCentimetrosDepoisAtaque)
  # print('diferenca %: ', diferencaPercentualDepoisAtaque)

  # adiciona na imagem o valor da área
  cv.putText(imgAntesRecortadaRedimensionada, 
            'Area total: {} cm2'.format(str(areaTotalAntesAtaque)), 
            (20, 60), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (0, 255, 255),1)

  # adiciona na imagem o valor da área
  cv.putText(imgDepoisRecortadaRedimensionada, 
            'Area total: {} cm2'.format(str(areaTotalDepoisAtaque)), 
            (20, 60), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (0, 255, 255),1)

  # coloca as imagens numa mesma janela
  imagensHorizontal = np.hstack((imgAntesRecortadaRedimensionada, imgDepoisRecortadaRedimensionada))

  # adicionar o resultado em cm2 na imagem
  cv.putText(imagensHorizontal, 
            'Diferenca em cm2: {} cm2'.format(str(diferencaCentimetrosDepoisAtaque)), 
            (20, 80), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (0, 255, 255),1)
  
  # adicionar o resultado em % na imagem
  cv.putText(imagensHorizontal, 
            'Diferenca em %: {} %'.format(str(diferencaPercentualDepoisAtaque)), 
            (20, 100), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (0, 255, 255),1)

  # mostra a imagem
  cv.imshow("Imagens", imagensHorizontal)

  key = cv.waitKey(0)

  cv.destroyAllWindows() 
# <><><><><><><><><><><><><><><><><><><><><><><><><><><>

