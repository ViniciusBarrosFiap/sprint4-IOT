import os, io, pickle, base64, numpy as np, dlib, cv2
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image


#Setup
PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
THRESH = 0.6  # tolerância (quanto menor, mais rígido)


app = FastAPI(title="API de Reconhecimento Facial (PKL compatível)")
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR)
recognizer = dlib.face_recognition_model_v1(RECOG)

# Carrega o banco existente (se houver)
db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}


# #Funções principais
def extrair_vetor_facial(imagem):
    
    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)
    
    if len(faces) != 1:
        return None
    shape = shape_predictor(rgb, faces[0])
    chip = dlib.get_face_chip(rgb, shape)
    vetor = np.array(recognizer.compute_face_descriptor(chip), dtype=np.float32)
    return vetor
#Converte uma imagem em base64 para formato OpenCV
def base64_para_cv2(base64_str):
    dados = base64.b64decode(base64_str)
    imagem = np.frombuffer(dados, np.uint8)
    return cv2.imdecode(imagem, cv2.IMREAD_COLOR)

def salvar_db():
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


#ENDPOINT: CADASTRAR ROSTO
@app.post("/cadastrar")
async def cadastrar_rosto(
    nome: str = Form(...),
    imagem: UploadFile = None,
    imagem_base64: str = Form(None)
):
    if not nome.strip():
        return JSONResponse({"erro": "Nome não informado"}, status_code=400)

    # Carregar imagem (Arquivos ou Base64)
    if imagem:
        conteudo = await imagem.read()
        npimg = np.frombuffer(conteudo, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    elif imagem_base64:
        frame = base64_para_cv2(imagem_base64)
    else:
        return JSONResponse({"erro": "Nenhuma imagem enviada"}, status_code=400)

    vetor = extrair_vetor_facial(frame)
    if vetor is None:
        return JSONResponse({"erro": "Rosto não detectado ou múltiplos rostos na imagem"}, status_code=400)

    db[nome] = vetor
    salvar_db()
    return {"mensagem": f"Usuário '{nome}' cadastrado com sucesso."}



# ENDPOINT: AUTENTICAR ROSTO
@app.post("/autenticar")
async def autenticar_rosto(
    imagem: UploadFile = None,
    imagem_base64: str = Form(None)
):
    if not db:
        return JSONResponse({"erro": "Banco de rostos vazio"}, status_code=400)

    # Carregar imagem
    if imagem:
        conteudo = await imagem.read()
        npimg = np.frombuffer(conteudo, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    elif imagem_base64:
        frame = base64_para_cv2(imagem_base64)
    else:
        return JSONResponse({"erro": "Nenhuma imagem enviada"}, status_code=400)

    vetor = extrair_vetor_facial(frame)
    if vetor is None:
        return JSONResponse({"erro": "Nenhum rosto detectado"}, status_code=400)

    # Comparar com o banco
    nome_encontrado, menor_dist = "Desconhecido", 999
    for nome, v in db.items():
        dist = np.linalg.norm(vetor - v)
        if dist < menor_dist:
            nome_encontrado, menor_dist = nome, dist

    autenticado = bool(menor_dist <= THRESH)
    return {
        "autenticado": autenticado,
        "usuario": nome_encontrado if autenticado else None,
        "distancia": float(round(menor_dist, 4))
    }
