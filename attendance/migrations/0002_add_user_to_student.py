from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('attendance', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='student',
            name='user',
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='student_profile', to='auth.user'),
        ),
    ] 