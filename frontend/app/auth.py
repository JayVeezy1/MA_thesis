import logging

import bcrypt
from flask import url_for, request
from flask_login import LoginManager
from itsdangerous import URLSafeTimedSerializer
from werkzeug.utils import redirect

# from app.blueprints.util import redirect_url
from frontend.app.model import User

# Login manager
login_mngr = LoginManager()
login_mngr.login_view = 'auth.login'
# login_mngr.login_message = 'Hello, you need login'    # flashed message
login_mngr.session_protection = "basic"

# Logging
log = logging.getLogger()


@login_mngr.user_loader
def load_user(session_token):
    return User.query.filter_by(session_token=session_token).first()


@login_mngr.unauthorized_handler
def unauthorized():
    return redirect(url_for('auth.login', next=url_for(request.endpoint), info_modal_title="Unauthorized",
                            info_modal_body="You need to be logged in."))


def get_hashed_password(plain_text_password: str or bytes) -> bytes:
    """
    Hash a password for the first time (using bcrypt, the salt is saved into the hash itself).
    :param plain_text_password: Password in plain text (str or unicode)
    :return: Hashed password
    """
    # Encode if ordinary string
    if isinstance(plain_text_password, str):
        plain_text_password = plain_text_password.encode('utf-8')
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def verify_password(plain_text_password: str or bytes, hashed_password: str or bytes) -> bool:
    """
    Check hashed password.
    :param plain_text_password:
    :param hashed_password:
    :return: True if the passwords match
    """
    # Encode if ordinary string
    if isinstance(plain_text_password, str):
        plain_text_password = plain_text_password.encode('utf-8')
    return bcrypt.checkpw(plain_text_password, hashed_password)


def generate_token(email, secret_key, salt):
    serializer = URLSafeTimedSerializer(secret_key)
    return serializer.dumps(email, salt=salt)


def confirm_token(token, secret_key, salt, expiration=3600):
    serializer = URLSafeTimedSerializer(secret_key)
    return serializer.loads(token, salt=salt, max_age=expiration)
