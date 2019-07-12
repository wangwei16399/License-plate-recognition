from django.db import models
import django.utils.timezone as timezone
import datetime
# Create your models here.
class Car(models.Model):
    cname = models.CharField(verbose_name='车牌',max_length=32,null=False,primary_key=True)
    statime = models.DateTimeField('入库时间',default=timezone.now)
    endtime = models.DateTimeField('出库时间',default=datetime.datetime(1970, 1, 1, 0, 0, 0))
    costs = models.FloatField('累计消费',default=0)

    def __str__(self):
        return self.cname