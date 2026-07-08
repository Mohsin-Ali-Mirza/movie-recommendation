import asyncio
import aiohttp
import random
import sys

MIDDLEWARE_URL = "http://localhost:8000"

async def request_training(worker_id, session):
    """Send a request to the middleware for training asynchronously."""
    async with session.post(f"{MIDDLEWARE_URL}/train") as response:
        result = await response.json()
        print(f"Worker {worker_id}: {result['message']}")
        sys.stdout.flush()  # Ensure log is printed immediately
        return result

async def worker(worker_id):
    """Simulate a user sending a training request asynchronously."""
    print(f"Worker {worker_id} started.")
    sys.stdout.flush()  # Ensure log is printed immediately

    async with aiohttp.ClientSession() as session:
        response = await request_training(worker_id, session)

        if "Training already in progress" in response["message"]:
            print(f"Worker {worker_id}: Using old model for inference. The Champion_ID: {response['champion_run_id']} The Challenger_ID: {response['challenger_run_id']}")
            sys.stdout.flush()  # Ensure log is printed immediately
        else:
            print(f"Worker {worker_id}: Training started.")
            sys.stdout.flush()  # Ensure log is printed immediately

    print(f"Worker {worker_id} finished. The new Champion_ID: {response['champion_run_id']} The new Challenger_ID: {response['challenger_run_id']}")
    sys.stdout.flush()  # Ensure log is printed immediately
    
async def main():
    """Run 10 workers asynchronously."""
    NUM_WORKERS = 20
    tasks = []

    for i in range(NUM_WORKERS):
        tasks.append(asyncio.create_task(worker(i)))
        await asyncio.sleep(random.uniform(0.5, 2))  # Simulate users sending requests at different times

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
