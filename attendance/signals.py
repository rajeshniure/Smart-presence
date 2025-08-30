from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.management import call_command
from django.conf import settings
import threading
import logging
import os

from .models import Student

# Setup logging
logger = logging.getLogger(__name__)

# Flag to prevent recursive retraining
_is_retraining = False

def retrain_facenet_async():
    """Retrain FaceNet model asynchronously to avoid blocking the request"""
    global _is_retraining
    
    try:
        logger.info("Starting automatic FaceNet retraining...")
        
        # Call the management command
        call_command('retrain_facenet', '--force', verbosity=1)
        
        logger.info("Automatic FaceNet retraining completed successfully!")
        
        # Reload models in the face pipeline
        try:
            from .utils.face_pipeline import reload_models
            reload_models()
            logger.info("Models reloaded successfully after retraining!")
        except Exception as e:
            logger.error(f"Error reloading models: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error during automatic FaceNet retraining: {str(e)}")
    finally:
        _is_retraining = False

@receiver(post_save, sender=Student)
def auto_retrain_on_student_save(sender, instance, created, **kwargs):
    """Automatically rebuild recognition index when a student is created/verified/image-updated.

    We trigger when:
    - Student is newly created with an image
    - Student is verified (is_verified=True) and has an image
    - Student's image field is updated
    """
    global _is_retraining

    should_trigger = False

    if created and instance.image:
        should_trigger = True
    else:
        update_fields = kwargs.get('update_fields')
        if update_fields is not None:
            if 'image' in update_fields or 'is_verified' in update_fields:
                should_trigger = True
        else:
            # update_fields not provided; if student is verified and has an image, retrain
            if instance.is_verified and instance.image:
                should_trigger = True

    if should_trigger and not _is_retraining:
        _is_retraining = True
        thread = threading.Thread(target=retrain_facenet_async)
        thread.daemon = True
        thread.start()

@receiver(post_delete, sender=Student)
def auto_retrain_on_student_delete(sender, instance, **kwargs):
    """Automatically retrain FaceNet model when a student is deleted"""
    global _is_retraining
    
    if not _is_retraining:
        _is_retraining = True
        
        logger.info(f"Student '{instance.name}' deleted. Triggering auto-retrain...")
        
        # Start retraining in a separate thread
        thread = threading.Thread(target=retrain_facenet_async)
        thread.daemon = True
        thread.start() 