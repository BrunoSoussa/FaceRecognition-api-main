from flask import Flask, request, jsonify
import cv2
import os
from model.prediction import FaceRecognition

app = Flask(__name__)

# Inicializar a classe FaceRecognition
model_path = r"model/keras/facenet_keras.h5"  # Substitua pelo caminho do modelo
collection_name = "face_embeddings"
chroma_path = "chroma_data"
face_recognition = FaceRecognition(model_path, collection_name, chroma_path)

# Rota para adicionar um rosto ao banco de dados
@app.route('/add_face', methods=['POST'])
def add_face():
    if 'image' not in request.files or 'identifier' not in request.form:
        return jsonify({"error": "Imagem e nome são obrigatórios"}), 400

    image_file = request.files['image']
    identifier = request.form['identifier']

    # Salvar a imagem temporariamente
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        # Adicionar o rosto ao banco de dados
        face_recognition.add_to_db(temp_image_path, identifier)
        os.remove(temp_image_path)
        return jsonify({"message": "Rosto adicionado com sucesso", "identifier": identifier}), 200
    except Exception as e:
        os.remove(temp_image_path)
        return jsonify({"error": str(e)}), 500

# Rota para reconhecer um rosto
@app.route('/find_face', methods=['POST'])
def find_face():
    if 'image' not in request.files:
        return jsonify({"error": "Imagem é obrigatória"}), 400

    image_file = request.files['image']

    # Salvar a imagem temporariamente
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        # Procurar rosto semelhante no banco de dados
        best_match = face_recognition.find_in_db(temp_image_path, threshold=0.5)
        os.remove(temp_image_path)

        if best_match:
            return jsonify({"message": "Rosto encontrado", "identifier": best_match}), 200
        else:
            return jsonify({"message": "Rosto não encontrado", "identifier": "Desconhecido"}), 404
    except Exception as e:
        os.remove(temp_image_path)
        return jsonify({"error": str(e)}), 500
    
@app.route('/add_face_from_video', methods=['POST'])
def add_face_from_video():
    if 'video' not in request.files or 'identifier' not in request.form:
        return jsonify({"error": "Vídeo e identifier são obrigatórios"}), 400

    video_file = request.files['video']
    identifier = request.form['identifier']

    # Salvar o vídeo temporariamente
    temp_video_path = "temp_video.mp4"
    video_file.save(temp_video_path)

    try:
        # Abrir o vídeo com OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = 0
        embeddings = []
        frame_skip = 10  # Pular 10 frames entre processamentos

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Pular frames para evitar redundância
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Converter o quadro para RGB (o OpenCV usa BGR por padrão)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extrair o rosto do quadro
            face = face_recognition._extract_face(rgb_frame)
            if face is not None:
                # Obter embedding do rosto
                embedding = face_recognition._get_embedding(face)
                embeddings.append(embedding)

            frame_count += 1
            if frame_count > 50: 
                break

        cap.release()
        os.remove(temp_video_path)

        # Adicionar os embeddings ao banco de dados
        if embeddings:
            for embedding in embeddings:
                face_recognition._collection.add(
                    embeddings=[embedding.tolist()],
                    ids=[identifier],
                    metadatas=[{"identifier": identifier}]
                )
            return jsonify({"message": f"Rostos adicionados com sucesso para {identifier}", "count": len(embeddings)}), 200
        else:
            return jsonify({"message": "Nenhum rosto detectado no vídeo"}), 404

    except Exception as e:
        os.remove(temp_video_path)
        return jsonify({"error": str(e)}), 500


@app.route('/find_face_from_video', methods=['POST'])
def find_face_from_video():
    if 'video' not in request.files:
        return jsonify({"error": "Vídeo é obrigatório"}), 400

    video_file = request.files['video']

    # Salvar o vídeo temporariamente
    temp_video_path = "temp_video.mp4"
    video_file.save(temp_video_path)

    try:
        # Abrir o vídeo com OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = 0
        results = {}
        frame_skip = 10  # Pular 10 frames entre processamentos

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Pular frames para evitar redundância
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Converter o quadro para RGB (o OpenCV usa BGR por padrão)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extrair o rosto do quadro
            face = face_recognition._extract_face(rgb_frame)
            if face is not None:
                # Obter embedding do rosto
                embedding = face_recognition._get_embedding(face)

                # Consultar o banco de dados com o embedding
                query_result = face_recognition._collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=1
                )

                # Verificar se o resultado atende ao limiar
                if query_result["distances"] and query_result["distances"][0][0] <= 0.5:
                    best_match = query_result["metadatas"][0][0]["name"]
                    similarity = 1 - query_result["distances"][0][0]

                    # Adicionar ao dicionário de resultados
                    if best_match in results:
                        results[best_match] += 1
                    else:
                        results[best_match] = 1

            frame_count += 1
            if frame_count > 50:  # Limitar para processar até 150 quadros no total
                break

        cap.release()
        os.remove(temp_video_path)

        # Processar os resultados
        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            return jsonify({
                "message": "Rostos encontrados",
                "results": [{"name": name, "count": count} for name, count in sorted_results]
            }), 200
        else:
            return jsonify({"message": "Nenhum rosto encontrado"}), 404

    except Exception as e:
        os.remove(temp_video_path)
        return jsonify({"error": str(e)}), 500
    
@app.route('/compare_face', methods=['POST'])
def compare_face():
    if 'image' not in request.files or 'identifier' not in request.form:
        return jsonify({"error": "Imagem e nome são obrigatórios"}), 400

    image_file = request.files['image']
    identifier = request.form['identifier']

    # Salvar a imagem temporariamente
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        # Obter embedding da imagem fornecida
        image = cv2.imread(temp_image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = face_recognition._extract_face(rgb_image)

        if face is None:
            os.remove(temp_image_path)
            return jsonify({"error": "Nenhum rosto detectado na imagem"}), 400

        embedding = face_recognition._get_embedding(face)

        # Consultar o banco de dados pelo nome fornecido
        query_result = face_recognition._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1,
            where={"name": identifier}
        )

        os.remove(temp_image_path)
        similarity = 1 - query_result["distances"][0][0]
        # Verificar se há correspondência e se atende ao limiar
        if query_result["distances"] and query_result["distances"][0][0] <= 0.5:
            
            return jsonify({
                "message": True,
                "identifier": identifier,
                "similarity": similarity,
            }), 200
        else:
            return jsonify({
                "message": False,
                "identifier": identifier,
                "similarity": similarity
            }), 404

    except Exception as e:
        os.remove(temp_image_path)
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
