# Generated by Django 4.1.2 on 2022-11-30 22:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Cheque',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_original', models.ImageField(null=True, upload_to='original')),
                ('image_preprocessing', models.ImageField(null=True, upload_to='preprocessing')),
                ('image_with_bounding_boxes', models.ImageField(null=True, upload_to='bounding_boxes')),
                ('image_issueBank', models.ImageField(null=True, upload_to='issueBank')),
                ('image_Rnom', models.ImageField(null=True, upload_to='Rnom')),
                ('image_montant_chiffre', models.ImageField(null=True, upload_to='montant_chiffre')),
                ('image_montant_lettre', models.ImageField(null=True, upload_to='montant_lettre')),
                ('image_cheque_number', models.ImageField(null=True, upload_to='cheque_number')),
                ('image_date', models.ImageField(null=True, upload_to='date')),
                ('image_account_number', models.ImageField(null=True, upload_to='account_number')),
                ('image_signature', models.ImageField(null=True, upload_to='signature')),
            ],
            options={
                'db_table': 'Cheque',
            },
        ),
    ]
