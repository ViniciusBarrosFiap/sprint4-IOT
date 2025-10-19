# API de Reconhecimento Facial - FastAPI + Dlib

# Integrantes
# Integrantes

- RM99499 – Guilherme Monteiro Espim
- RM99279 – João Paulo Fonseca Zamperlini
- RM98297 – Lucas Pisaneschi Speranzini
- RM97937 – Pedro Henrique Fernandes Lô de Barros
- RM97824 – Vinicius Oliveira de Barros

## Sinopse

Este projeto implementa uma API de **reconhecimento facial** utilizando **FastAPI**, **Dlib** e **OpenCV**, com foco em **cadastro** e **autenticação facial** através de imagens. Os vetores faciais extraídos são armazenados localmente em um banco `db.pkl`, possibilitando o uso local e offline.

---

## ⚙️ Funcionalidade

| Endpoint | Método | Descrição |
|-----------|---------|------------|
| `/cadastrar` | `POST` | Cadastra um novo rosto com nome e imagem. |
| `/autenticar` | `POST` | Compara uma imagem com o banco existente e retorna o usuário correspondente. |

---

## Video demonstrativo

```bash
https://www.youtube.com/watch?v=15VV0_NmKWk
```

## Objetivos do Projeto

* Criar endpoints para cadastro e autenticação facial
* Extrair vetores faciais com Dlib a partir de imagens
* Armazenar vetores faciais localmente (Pickle)
* Suportar envio de imagem via upload ou base64
* Implementar lógica de comparação facial
* Documentar API com Swagger (FastAPI Docs)

---

## 1. Estrutura e Objetivo Geral

* **FastAPI**: Framework leve e rápido para criação da API
* **Dlib + OpenCV**: Extração e manipulação de vetores faciais
* **Banco de vetores (Pickle)**: Persistência dos rostos cadastrados
* **Endpoints REST**: Cadastro e autenticação por imagem

**Comando de validação da estrutura:**

```bash
ls -lh servidor_face.py db.pkl *.dat
```


---

## 2. Instalação e Execução da API

**Instalação das dependências:**

```bash
pip install fastapi uvicorn dlib opencv-python numpy pillow python-multipart
```

**Executar o servidor local:**

```bash
uvicorn servidor_face:app --reload
```

**Acessar a documentação Swagger:**

```bash
http://127.0.0.1:8000/docs
```


---

## 3. Cadastro Facial - Endpoint `/cadastrar`

Permite cadastrar um novo usuário fornecendo nome e imagem (upload ou base64).

**Comando de teste:**

```bash
curl -X POST http://127.0.0.1:8000/cadastrar \
  -F "nome=Vinicius" \
  -F "imagem_base64=$(base64 -w0 vinicius.jpg)"
```

**Retorno esperado:**

```json
{ "mensagem": "Usuário 'Vinicius' cadastrado com sucesso." }
```

![Cadastro facial realizado](./imagens/cadastro.png)

---

## 4. Autenticação Facial - Endpoint `/autenticar`

Compara uma imagem recebida com os rostos salvos no banco `db.pkl`.

**Comando de teste:**

```bash
curl -X POST http://127.0.0.1:8000/autenticar \
  -F "imagem_base64=$(base64 -w0 teste.jpg)"
```

**Retorno esperado:**

```json
{
  "autenticado": true,
  "usuario": "Vinicius",
  "distancia": 0.4123
}
```

---

## 5. Funcionamento Interno

* Conversão da imagem base64 → OpenCV
* Detecção facial com `dlib.get_frontal_face_detector()`
* Vetor facial extraído com `compute_face_descriptor()`
* Comparação via distância Euclidiana (`np.linalg.norm()`)
* Autenticado se distância ≤ 0.6

**Visualização de vetores:**

```python
{ "Vinicius": [0.12, -0.34, 0.56, ...] }
```


---

## 6. Validação e Logs

Todas as requisições são logadas no terminal. Erros de envio (formato inválido, ausência de face) são tratados com códigos HTTP adequados.

**Exemplo de erro tratado:**

```json
{ "erro": "Nenhum rosto detectado na imagem" }
```

---

## Documentação

O projeto está documentado com:

* Dependências e execução no `README.md`
* Swagger (`/docs`) com teste de endpoints
* Prints de teste para cada funcionalidade

---

## ✅ Conclusão

Este projeto demonstra a viabilidade de uma API leve para **reconhecimento facial local**, aplicando:

* Vetorização de rostos com Dlib
* Comparação offline com banco local
* Suporte a base64 e upload
* Integração com aplicativos mobile ou web

Ideal para soluções que exigem **privacidade**, **baixa latência** e **autenticação biométrica simplificada**.
