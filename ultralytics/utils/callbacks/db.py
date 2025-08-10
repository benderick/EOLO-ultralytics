# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers
import os

try:
    assert not TESTS_RUNNING  # do not log pytest
    from ultralytics.utils import ROOT
    
    import datetime
    import peewee
    assert hasattr(peewee, "__version__")
    db_path = str(ROOT / "logs" / 'db.db')
    assert os.path.exists(db_path)
    db = peewee.SqliteDatabase(db_path)
    
    class EOLO_RUN(peewee.Model):
        id = peewee.IntegerField(primary_key=True)
        project = peewee.CharField(null=True)
        name = peewee.CharField(null=True)
        base_model = peewee.CharField(null=True)
        scale = peewee.CharField(null=True)
        data = peewee.CharField(null=True)
        group = peewee.CharField(null=True)
        notes = peewee.CharField(null=True)
        location = peewee.CharField(null=True)
        tags = peewee.CharField(null=True)
        map = peewee.TextField(default="[]")
        map50 = peewee.TextField(default="[]")
        is_basic = peewee.BooleanField(default=False)
        wb = peewee.CharField(null=True)
        created = peewee.DateTimeField(default=datetime.datetime.now) 
        info = peewee.TextField(null=True)
        exp_timestamp = peewee.CharField(null=True)

        class Meta:
            database = db
            db_table = 'eolo_runs'

except (ImportError, AssertionError):
    db = None

run = None

def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    logger_info = trainer.logger
    global run
    run = EOLO_RUN.create(location=str(trainer.save_dir.parent), **logger_info)
    run.save()


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    
    if trainer.epoch == 0:
        run.info = str(model_info_for_loggers(trainer))
        try:
            import wandb as wb
            if wb.run:
                run.wb = wb.run.id
                run.save()
        except:
            pass
    
    # 记录模型的mAP和mAP50到数据库
    run.map = str(eval(run.map) + [trainer.metrics["metrics/mAP50-95(B)"]])
    run.map50 = str(eval(run.map50) + [trainer.metrics["metrics/mAP50(B)"]])
    run.save()
    return # 暂停低指标中断
    # 对于非基本模型，在低于基本模型的map和map50时停止训练
    if not run.is_basic:
        idx = epoche_list.index(trainer.epoch)
        basics = Run.select().where((Run.project == run.project) & Run.is_basic)
        true_basics_map = []
        true_basics_map50 = []
        try:
            for basic in basics:
                if set(eval(basic.tags)) & set(eval(run.tags)):
                    true_basics_map.append(eval(basic.map)[idx])    
                    true_basics_map50.append(eval(basic.map50)[idx])  
        except:
            print("在获取基本模型的map和map50时发生错误")
        if len(true_basics_map) > 0 and len(true_basics_map50) > 0:
            run_map = trainer.metrics["metrics/mAP50-95(B)"]
            run_map50 = trainer.metrics["metrics/mAP50(B)"]
            for map in true_basics_map:
                if run_map < map * 0.95:
                    trainer.need_to_finish = True
                    break
            for map50 in true_basics_map50:
                if run_map50 < map50 * 0.9:
                    trainer.need_to_finish = True
                    break
            if trainer.need_to_finish:
                print("触发低指标中断")
                wb.run.alert(title="低指标中断",
                            text=f"Run {run.id} was killed because of low metrics in epoch {trainer.epoch}. \n In project {run.project}, name {run.name}, tags {run.tags}.",
                            level=wb.AlertLevel.WARN,
                            wait_duration=300,)



callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if db
    else {}
)
