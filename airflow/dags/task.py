from airflow.sdk import dag, task
from pendulum import datetime


@dag(
    dag_id="orders_import_etl",
    schedule=None,
    start_date=datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["intro", "taskflow"],
)
def intro_taskflow_etl():
    @task
    def extract():
        return {"orders": [100, 250, 175, 90]}

    @task
    def transform(data):
        orders = data["orders"]
        total = sum(orders)
        average = total / len(orders)
        return {"total": total, "average": average, "count": len(orders)}

    @task
    def load(summary):
        print(f"Order count: {summary['count']}")
        print(f"Total sales: {summary['total']}")
        print(f"Average sales per order: {summary['average']}")

    @task.bash
    def create_dag_log():
        return "touch /home/repl/orders_executed.log"

    load(transform(extract()))
    create_dag_log()


intro_taskflow_etl()
