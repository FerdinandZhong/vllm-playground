#!/usr/bin/env python3
"""
Example: Deploy vLLM on Ray Cluster

This example shows how to connect vLLM Playground to an existing Ray cluster
and deploy a model using Ray Serve.

Prerequisites:
1. Ray cluster running (started via launch_cluster.py)
2. Ray address available (e.g., 127.0.0.1:6379)

Usage:
    # Set Ray address (if not auto-detected)
    export RAY_ADDRESS=127.0.0.1:6379

    # Run the example
    python examples/ray_deployment_example.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def deploy_vllm_on_ray():
    """Deploy vLLM model on existing Ray cluster."""

    from vllm_playground.ray_serve import ray_backend

    print("=" * 60)
    print("vLLM Playground - Ray Deployment Example")
    print("=" * 60)

    # Step 1: Check if Ray address is set
    ray_address = os.environ.get('RAY_ADDRESS')
    if ray_address:
        print(f"\n✓ Using Ray address from environment: {ray_address}")
    else:
        print("\n⚠ RAY_ADDRESS not set - will auto-detect or start local cluster")
        ray_address = None

    # Step 2: Initialize Ray connection
    print("\n[1/4] Connecting to Ray cluster...")
    init_result = await ray_backend.initialize_ray(address=ray_address)

    if init_result['initialized']:
        print(f"    ✓ Connected successfully")
        print(f"    Mode: {init_result['mode']}")
        print(f"    Address: {init_result['address']}")
        print(f"    Nodes: {init_result['nodes']}")
    else:
        print(f"    ✗ Connection failed: {init_result.get('message')}")
        return

    # Step 3: Configure vLLM deployment
    print("\n[2/4] Configuring vLLM deployment...")

    vllm_config = {
        # Model configuration
        'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Small model for testing

        # Resource configuration
        'tensor_parallel_size': 1,  # Single GPU (or CPU)
        'use_cpu': False,  # Set to True for CPU-only mode

        # Server configuration
        'port': 8000,

        # Performance tuning
        'gpu_memory_utilization': 0.9,
        'dtype': 'auto',

        # Optional: Set to True for gated models
        'trust_remote_code': False,
    }

    print(f"    Model: {vllm_config['model']}")
    print(f"    Port: {vllm_config['port']}")
    print(f"    Tensor Parallel Size: {vllm_config['tensor_parallel_size']}")
    print(f"    CPU Mode: {vllm_config['use_cpu']}")

    # Step 4: Deploy the model
    print("\n[3/4] Deploying vLLM model...")
    print("    (This may take a few minutes to download and load the model)")

    result = await ray_backend.start_model(
        vllm_config=vllm_config,
        wait_ready=True  # Wait until model is ready
    )

    if result['started']:
        print(f"    ✓ Model deployed successfully")
        print(f"    Application: {result['application_name']}")
        print(f"    Status: {result['status']}")
        if result.get('ready'):
            print(f"    Ready: Yes (took {result.get('elapsed_time')}s)")
    else:
        print(f"    ✗ Deployment failed: {result.get('message')}")
        return

    # Step 5: Test the deployment
    print("\n[4/4] Testing the deployment...")

    try:
        import aiohttp

        # Test health endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{vllm_config['port']}/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"    ✓ Health check passed")
                    print(f"    Engine: {health_data.get('engine')}")
                    print(f"    Model: {health_data.get('model')}")
                else:
                    print(f"    ✗ Health check failed: HTTP {resp.status}")

        # Test completion endpoint
        print("\n    Testing text completion...")
        async with aiohttp.ClientSession() as session:
            completion_request = {
                "model": vllm_config['model'],
                "prompt": "Once upon a time",
                "max_tokens": 50,
                "temperature": 0.7
            }

            async with session.post(
                f"http://localhost:{vllm_config['port']}/v1/completions",
                json=completion_request
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    generated_text = data['choices'][0]['text']
                    print(f"    ✓ Completion successful")
                    print(f"    Generated: {generated_text[:100]}...")
                else:
                    print(f"    ✗ Completion failed: HTTP {resp.status}")

    except ImportError:
        print("    ⚠ aiohttp not available - skipping API tests")
        print("    Install with: pip install aiohttp")

    # Summary
    print("\n" + "=" * 60)
    print("Deployment Summary")
    print("=" * 60)
    print(f"Model deployed and ready at: http://localhost:{vllm_config['port']}")
    print(f"\nOpenAI-compatible endpoints:")
    print(f"  - Completions:      POST http://localhost:{vllm_config['port']}/v1/completions")
    print(f"  - Chat Completions: POST http://localhost:{vllm_config['port']}/v1/chat/completions")
    print(f"  - Models:           GET  http://localhost:{vllm_config['port']}/v1/models")
    print(f"  - Health:           GET  http://localhost:{vllm_config['port']}/health")

    print(f"\nTest with curl:")
    print(f"""  curl -X POST http://localhost:{vllm_config['port']}/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{{
      "model": "{vllm_config['model']}",
      "prompt": "Hello, how are you?",
      "max_tokens": 50
    }}'""")

    print(f"\nTest with OpenAI Python client:")
    print(f"""  from openai import OpenAI
  client = OpenAI(
      base_url="http://localhost:{vllm_config['port']}/v1",
      api_key="dummy"  # vLLM doesn't require real API key
  )

  response = client.completions.create(
      model="{vllm_config['model']}",
      prompt="Hello, how are you?",
      max_tokens=50
  )
  print(response.choices[0].text)""")

    print("\n" + "=" * 60)
    print("To stop the deployment:")
    print("  python -c 'import asyncio; from vllm_playground.ray_serve import ray_backend; asyncio.run(ray_backend.stop_model())'")
    print("=" * 60)


async def check_deployment_status():
    """Check status of existing deployment."""
    from vllm_playground.ray_serve import ray_backend

    print("\nChecking deployment status...")
    status = await ray_backend.get_status()

    print(f"Running: {status.get('running')}")
    print(f"Status: {status.get('status')}")

    if status.get('deployments'):
        print("\nDeployments:")
        for dep in status['deployments']:
            print(f"  - {dep['name']}: {dep['status']}")


async def stop_deployment():
    """Stop the deployment."""
    from vllm_playground.ray_serve import ray_backend

    print("\nStopping deployment...")
    result = await ray_backend.stop_model()

    if result['stopped']:
        print(f"✓ Deployment stopped: {result['status']}")
    else:
        print(f"✗ Failed to stop: {result.get('error')}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Deploy vLLM on Ray cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy model (auto-detect Ray cluster)
  python examples/ray_deployment_example.py

  # Deploy with specific Ray address
  RAY_ADDRESS=192.168.1.100:6379 python examples/ray_deployment_example.py

  # Check deployment status
  python examples/ray_deployment_example.py --status

  # Stop deployment
  python examples/ray_deployment_example.py --stop
        """
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check deployment status'
    )

    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop deployment'
    )

    args = parser.parse_args()

    try:
        if args.status:
            asyncio.run(check_deployment_status())
        elif args.stop:
            asyncio.run(stop_deployment())
        else:
            asyncio.run(deploy_vllm_on_ray())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
