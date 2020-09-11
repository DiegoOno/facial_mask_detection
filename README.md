## Detecção de máscaras com Redes Neurais Convolucionais (CNN)

## Objetivo
Realizar a detecção de máscaras faciais, com a combinação de um classificador baseado em Redes Neurais Convolucionais, e um detector utilizando o método Haar Cascade.

### Base de dados
A base de dados contém 1500 imagens divididas em três categorias: Treinamento, Validação e Teste;

A divisão detalhada consta na tabela a seguir:

|    Classe   | Treinamento | Validação | Teste | Total |
|:-----------:|:-----------:|:---------:|:-----:|:-----:|
| Com máscara |     1000    |    400    |  100  |  1500 |
| Sem máscara |     1000    |    400    |  100  |  1500 |

Link da base de dados:

https://drive.google.com/file/d/1cwarWkpgLmEHKfv8ncjjzEYPTCHt502G/view?usp=sharing

### Treinamento da CNN
O treinamento do modelo é realizado pelo script  __mask_classificator_training.py__. 

Para executá-lo:
```sh
python3 mask_classificator_training.py
```

### Validação do modelo
O script de validação possui o nome de __validation.py__

Para executar a validação:
```sh
python3 validation.py -m [model_path/model_filename.hdf5]
```

### Predição de uma única imagem
O script de predição de uma única imagem possui o nome de __prediction.py__

Para executar a predição:
```sh
python3 prediction.py -m [model_path/model_filename.hdf5] -i [image_path/image_file]
```

### Detecção de uma única imagem
O script de detecção de uma única imagem possui o nome de __detection.py__

Para executar a detecção:
```sh
python3 detection.py -m [model_path/model_filename.hdf5] -i [image_path/image_file]
```

# TODO
- [ ] Implementar detecção em vídeo.

