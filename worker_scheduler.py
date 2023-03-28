import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# every 30 minutes
SCHEDULE = os.environ.get('CRON_SCHEDULE', '*/30 * * * *')

scheduler = BlockingScheduler()


def trigger():
    print('Running cron')
    print('Finished cron')

print(f'Worker with schedule: {SCHEDULE}')

scheduler.add_job(trigger, CronTrigger.from_crontab(SCHEDULE))
scheduler.start()
