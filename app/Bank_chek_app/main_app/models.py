from django.db import models

# Create your models here.
from PIL import Image
from django.db import models
import numpy as np
import cv2


# Create your models here.
class Cheque(models.Model):
    image_original = models.ImageField(upload_to="original", null=True)
    image_preprocessing = models.ImageField(upload_to="preprocessing", null=True)
    image_with_bounding_boxes = models.ImageField(upload_to="bounding_boxes", null=True)
    image_issueBank = models.ImageField(upload_to="issueBank", null=True)
    image_ReceiverName = models.ImageField(upload_to="ReceiverName", null=True)
    image_amount_digits = models.ImageField(upload_to="amount_digits", null=True)
    image_amout_letter = models.ImageField(upload_to="amout_letter", null=True)
    image_cheque_number = models.ImageField(upload_to="cheque_number", null=True)
    image_DateIss = models.ImageField(upload_to="DateIss", null=True)
    image_account_number = models.ImageField(upload_to="account_number", null=True)
    image_signature = models.ImageField(upload_to="signature", null=True)


    # def save(self, *args, **kwargs):
    #     super().save(*args, **kwargs)
    #     img = Image.open(self.image.path)
    #     img.show()
    #     # new_image = process_image(img)
    #
    #     return super().save(*args, **kwargs)

    class Meta:
        db_table = 'Cheque'

