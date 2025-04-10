from flask import Blueprint, request, jsonify
from app.services.attt_knowledge import search_materials
from app.services.research_helper import get_research_guide

main_bp = Blueprint('main', __name__)

@main_bp.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    
    # Xử lý message với model AI
    response = process_message(message)
    
    return jsonify({'response': response})

@main_bp.route('/api/search-materials', methods=['POST'])
def search_materials_route():
    subject = request.json.get('subject')
    results = search_materials(subject)
    return jsonify(results)

@main_bp.route('/api/research-guide', methods=['GET'])
def research_guide():
    guide_type = request.args.get('type')
    return jsonify(get_research_guide(guide_type))