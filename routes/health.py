"""
Health check routes
"""
from config import Config
from datetime import datetime
from flask import jsonify, Blueprint
from services.products import get_product_count
from services.history import conversation_history
from web_database import get_web_db_connection, release_web_db_connection
from auth_database import get_auth_db_connection, release_auth_db_connection
from telegram_database import get_telegram_db_connection, release_telegram_db_connection

health_bp = Blueprint('health', __name__)

@health_bp.route("/health")
def health_check():
    """Health check endpoint for Railway"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "databases": {
            "telegram": "not configured",
            "web": "not configured",
            "auth": "not configured"
        },
        "products_loaded": get_product_count(),
        "active_conversations": len(conversation_history),
        "platform": "Railway"
    }

    if Config.TELEGRAM_DATABASE_URL:
        conn = get_telegram_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                health["databases"]["telegram"] = "connected"
            except Exception:
                health["databases"]["telegram"] = "error"
                health["status"] = "unhealthy"
            finally:
                release_telegram_db_connection(conn)
        else:
            health["databases"]["telegram"] = "connection_failed"
            health["status"] = "degraded"

    if Config.WEB_DATABASE_URL:
        conn = get_web_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                health["databases"]["web"] = "connected"
            except Exception:
                health["databases"]["web"] = "error"
                health["status"] = "unhealthy"
            finally:
                release_web_db_connection(conn)
        else:
            health["databases"]["web"] = "connection_failed"
            health["status"] = "degraded"

    if Config.AUTH_DATABASE_URL:
        conn = get_auth_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                health["databases"]["auth"] = "connected"
            except Exception:
                health["databases"]["auth"] = "error"
                health["status"] = "unhealthy"
            finally:
                release_auth_db_connection(conn)
        else:
            health["databases"]["auth"] = "connection_failed"
            health["status"] = "degraded"
    
    status_code = 200 if health["status"] == "healthy" else 503
    return jsonify(health), status_code
