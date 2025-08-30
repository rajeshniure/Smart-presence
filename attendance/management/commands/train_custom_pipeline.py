from django.core.management.base import BaseCommand
from django.conf import settings
import os
import pickle


class Command(BaseCommand):
    help = "Summarize training artifacts and generate simple curves from history."

    def handle(self, *args, **options):
        models_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
        det_hist_path = os.path.join(models_dir, 'detection_training_history.pkl')
        rec_hist_path = os.path.join(models_dir, 'recognition_training_history.pkl')

        if os.path.exists(det_hist_path):
            self._plot_history(det_hist_path, 'detection')
        else:
            self.stdout.write('No detection history found')

        if os.path.exists(rec_hist_path):
            self._plot_history(rec_hist_path, 'recognition')
        else:
            self.stdout.write('No recognition history found')

    def _plot_history(self, path: str, name: str):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with open(path, 'rb') as f:
            hist = pickle.load(f)
        out_dir = os.path.dirname(path)
        # Loss curve
        if 'train_losses' in hist and hist['train_losses']:
            fig, ax = plt.subplots()
            ax.plot(hist['train_losses'], label='Train')
            if 'val_losses' in hist:
                ax.plot(hist.get('val_losses', []), label='Val')
                ax.legend()
            ax.set_title(f'{name.title()} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            fig.savefig(os.path.join(out_dir, f'{name}_train_loss.png'), bbox_inches='tight')
            plt.close(fig)
        # Accuracy curve
        if 'val_accuracies' in hist and hist['val_accuracies']:
            fig, ax = plt.subplots()
            if 'train_accuracies' in hist:
                ax.plot(hist.get('train_accuracies', []), label='Train')
            ax.plot(hist['val_accuracies'], label='Val')
            ax.legend()
            ax.set_title(f'{name.title()} Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            fig.savefig(os.path.join(out_dir, f'{name}_accuracy.png'), bbox_inches='tight')
            plt.close(fig)
        # Precision/Recall for train if available
        if 'train_precisions' in hist and 'train_recalls' in hist:
            fig, ax = plt.subplots()
            ax.plot(hist['train_precisions'], label='Train Precision')
            ax.plot(hist['train_recalls'], label='Train Recall')
            ax.legend()
            ax.set_title(f'{name.title()} Precision/Recall (Train)')
            ax.set_xlabel('Epoch')
            fig.savefig(os.path.join(out_dir, f'{name}_train_prec_rec.png'), bbox_inches='tight')
            plt.close(fig)
        # F1 curve (if present)
        if 'val_f1s' in hist and hist['val_f1s']:
            fig, ax = plt.subplots()
            if 'train_f1s' in hist:
                ax.plot(hist.get('train_f1s', []), label='Train')
            ax.plot(hist['val_f1s'], label='Val')
            ax.legend()
            ax.set_title(f'{name.title()} F1 Score')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1')
            fig.savefig(os.path.join(out_dir, f'{name}_f1.png'), bbox_inches='tight')
            plt.close(fig)
        # Precision/Recall (if present)
        if 'val_precisions' in hist and 'val_recalls' in hist:
            fig, ax = plt.subplots()
            ax.plot(hist['val_precisions'], label='Precision')
            ax.plot(hist['val_recalls'], label='Recall')
            ax.legend()
            ax.set_title(f'{name.title()} Precision/Recall (Val)')
            ax.set_xlabel('Epoch')
            fig.savefig(os.path.join(out_dir, f'{name}_prec_rec.png'), bbox_inches='tight')
            plt.close(fig)
        # Confusion matrices (last epoch), with optional class subsampling to avoid overcrowding
        if 'val_confusion_matrices' in hist and hist['val_confusion_matrices']:
            import seaborn as sns
            cm = hist['val_confusion_matrices'][-1]
            labels = hist.get('class_names', None)
            # Optionally reduce to top-N classes by support (diagonal of cm)
            N = 12
            if labels and len(labels) > N:
                import numpy as np
                diag = np.diag(cm)
                top_idx = np.argsort(diag)[-N:]
                cm = cm[np.ix_(top_idx, top_idx)]
                labels = [labels[i] for i in top_idx]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels if labels else True,
                        yticklabels=labels if labels else True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{name.title()} Validation Confusion Matrix')
            fig.savefig(os.path.join(out_dir, f'{name}_confusion_matrix.png'), bbox_inches='tight')
            plt.close(fig)

        if 'train_confusion_matrices' in hist and hist['train_confusion_matrices']:
            import seaborn as sns
            cm = hist['train_confusion_matrices'][-1]
            labels = hist.get('class_names', None)
            N = 12
            if labels and len(labels) > N:
                import numpy as np
                diag = np.diag(cm)
                top_idx = np.argsort(diag)[-N:]
                cm = cm[np.ix_(top_idx, top_idx)]
                labels = [labels[i] for i in top_idx]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=labels if labels else True,
                        yticklabels=labels if labels else True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{name.title()} Training Confusion Matrix')
            fig.savefig(os.path.join(out_dir, f'{name}_train_confusion_matrix.png'), bbox_inches='tight')
            plt.close(fig)
        self.stdout.write(self.style.SUCCESS(f'Generated plots for {name}'))


