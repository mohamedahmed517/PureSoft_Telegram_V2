"""
Web chat routes
"""
import base64
from utils.logger import logger
from utils.metrics import metrics
from services.gemini import gemini_chat
from flask import Blueprint, render_template, request, jsonify, session
from auth_database import authenticate_user, register_user, get_user_info
from web_database import load_web_conversation, save_web_conversation, clear_web_conversation

web_chat_bp = Blueprint('web_chat', __name__)

@web_chat_bp.route("/chat")
def chat_page():
    """Main chat interface page"""
    return render_template('chat.html')

@web_chat_bp.route("/api/auth/register", methods=["POST"])
def api_register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not email or not password:
            return jsonify({"success": False, "error": "All fields required"}), 400
        
        if len(password) < 6:
            return jsonify({"success": False, "error": "Password must be at least 6 characters"}), 400
        
        user_id, error = register_user(username, email, password)
        
        if error:
            return jsonify({"success": False, "error": error}), 400
        
        session['user_id'] = user_id
        session['username'] = username
        
        logger.info(f"✅ New user registered via web: {username}")
        return jsonify({"success": True, "user_id": user_id, "username": username}), 200
        
    except Exception as e:
        logger.error(f"❌ Registration error: {e}")
        return jsonify({"success": False, "error": "Registration failed"}), 500

@web_chat_bp.route("/api/auth/login", methods=["POST"])
def api_login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400
        
        user_id, error = authenticate_user(username, password)
        
        if error:
            return jsonify({"success": False, "error": error}), 401
        
        session['user_id'] = user_id
        session['username'] = username
        
        logger.info(f"✅ User logged in via web: {username}")
        return jsonify({"success": True, "user_id": user_id, "username": username}), 200
        
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        return jsonify({"success": False, "error": "Login failed"}), 500

@web_chat_bp.route("/api/auth/logout", methods=["POST"])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({"success": True}), 200

@web_chat_bp.route("/api/auth/me", methods=["GET"])
def api_me():
    """Get current user info"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    user_info = get_user_info(user_id)
    if not user_info:
        return jsonify({"success": False, "error": "User not found"}), 404
    
    return jsonify({"success": True, "user": user_info}), 200

@web_chat_bp.route("/api/chat/send", methods=["POST"])
def api_chat_send():
    """Send a message and get AI response"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "error": "Please login first"}), 401
        
        data = request.get_json()
        message = data.get('message', '').strip()
        image_b64 = data.get('image')
        
        if not message and not image_b64:
            return jsonify({"success": False, "error": "Message or image required"}), 400
        
        user_key = f"web:{user_id}"
        
        response = gemini_chat(
            text=message,
            image_b64=image_b64,
            user_key=user_key
        )
        
        metrics.track_message("web_chat")
        
        return jsonify({
            "success": True,
            "response": response
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Chat send error: {e}")
        metrics.track_error("web_chat")
        return jsonify({"success": False, "error": "Failed to process message"}), 500

@web_chat_bp.route("/api/chat/history", methods=["GET"])
def api_chat_history():
    """Get chat history"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "error": "Not authenticated"}), 401
        
        history = load_web_conversation(user_id)
        return jsonify({"success": True, "history": history}), 200
        
    except Exception as e:
        logger.error(f"❌ History load error: {e}")
        return jsonify({"success": False, "error": "Failed to load history"}), 500

@web_chat_bp.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    """Clear chat history"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "error": "Not authenticated"}), 401
        
        clear_web_conversation(user_id)
        
        return jsonify({"success": True}), 200
        
    except Exception as e:
        logger.error(f"❌ Clear chat error: {e}")
        return jsonify({"success": False, "error": "Failed to clear chat"}), 500
