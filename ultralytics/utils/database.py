# import datetime
# import peewee
# from ultralytics.utils import ROOT

# db = peewee.SqliteDatabase(str(ROOT / "logs" / 'db.db'))

# class Run(peewee.Model):
#     id = peewee.CharField(primary_key=True)
#     project = peewee.CharField(null=True)
#     name = peewee.CharField(null=True)
#     base_model = peewee.CharField(null=True)
#     scale = peewee.CharField(null=True)
#     data = peewee.CharField(null=True)
#     group = peewee.CharField(null=True)
#     notes = peewee.CharField(null=True)
#     location = peewee.CharField(null=True)
#     tags = peewee.CharField(null=True)
#     map = peewee.CharField(default="[]")
#     map50 = peewee.CharField(default="[]")
#     is_basic = peewee.BooleanField(default=False)
#     created = peewee.DateTimeField(default=datetime.datetime.now) 

#     class Meta:
#         database = db
#         db_table = 'runs'
        
# class EOLO_RUN(peewee.Model):
#         id = peewee.IntegerField(primary_key=True)
#         project = peewee.CharField(null=True)
#         name = peewee.CharField(null=True)
#         base_model = peewee.CharField(null=True)
#         scale = peewee.CharField(null=True)
#         data = peewee.CharField(null=True)
#         group = peewee.CharField(null=True)
#         notes = peewee.CharField(null=True)
#         location = peewee.CharField(null=True)
#         tags = peewee.CharField(null=True)
#         map = peewee.TextField(default="[]")
#         map50 = peewee.TextField(default="[]")
#         is_basic = peewee.BooleanField(default=False)
#         wb = peewee.CharField(null=True)
#         created = peewee.DateTimeField(default=datetime.datetime.now) 
#         info = peewee.TextField(null=True)
#         exp_timestamp = peewee.CharField(null=True)

#         class Meta:
#             database = db
#             db_table = 'eolo_runs'


# if __name__ == "__main__":
#     EOLO_RUN.create_table()
#     run1 = EOLO_RUN.create()
#     run1.save()
