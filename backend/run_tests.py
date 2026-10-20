import asyncio
import os
import sys
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Set PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_gemini_embeddings():
    print("Testing Gemini Embeddings...")
    try:
        from services.embedder import embed_query
        emb = embed_query("Python Developer")
        print(f"  [SUCCESS] Gemini embeddings work! Vector dimensions: {len(emb)}")
        return True
    except Exception as e:
        print(f"  [ERROR] Gemini embeddings failed: {e}")
        return False


async def test_groq():
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    print(f"Testing Groq ({model})...")
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            model=model,
        )
        print(f"  [SUCCESS] Groq works! Response: '{chat_completion.choices[0].message.content.strip()}'")
        return True
    except Exception as e:
        print(f"  [ERROR] Groq failed: {e}")
        return False


async def test_redis():
    print("Testing Upstash Redis Cache...")
    try:
        from services.cache import cache_jobs, get_cached_jobs
        test_jobs = [{"title": "Software Engineer Test", "company": "Test Inc"}]
        await cache_jobs(test_jobs, "test-query", "test-loc")
        cached = await get_cached_jobs("test-query", "test-loc")
        if cached == test_jobs:
            print("  [SUCCESS] Upstash Redis cache read/write works!")
            return True
        else:
            print(f"  [ERROR] Upstash Redis data mismatch: {cached}")
            return False
    except Exception as e:
        print(f"  [ERROR] Upstash Redis failed: {e}")
        return False


async def test_jsearch():
    print("Testing JSearch (RapidAPI)...")
    try:
        from services.agent import _search_jsearch
        jobs = await _search_jsearch("Python", "Dhaka")
        print(f"  [SUCCESS] JSearch returned {len(jobs)} jobs!")
        return True
    except Exception as e:
        print(f"  [ERROR] JSearch failed: {e}")
        return False


async def test_remotive():
    print("Testing Remotive API...")
    try:
        from services.agent import _search_remotive
        jobs = await _search_remotive("Python")
        print(f"  [SUCCESS] Remotive returned {len(jobs)} jobs!")
        return True
    except Exception as e:
        print(f"  [ERROR] Remotive failed: {e}")
        return False


async def test_tavily():
    print("Testing Tavily Search...")
    try:
        from services.agent import _search_tavily
        jobs = await _search_tavily("Python Developer", "Dhaka")
        print(f"  [SUCCESS] Tavily search returned {len(jobs)} jobs!")
        return True
    except Exception as e:
        print(f"  [ERROR] Tavily failed: {e}")
        return False


async def main():
    print("=" * 60)
    print("CareerPilot Services Integration Tests")
    print("=" * 60)
    
    results = {}
    results["Gemini Embeddings"] = await test_gemini_embeddings()
    results["Upstash Redis"] = await test_redis()
    results["Groq"] = await test_groq()
    results["JSearch"] = await test_jsearch()
    results["Remotive"] = await test_remotive()
    results["Tavily"] = await test_tavily()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    all_ok = True
    for service, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"{service:20}: {status_str}")
        if not status:
            all_ok = False
            
    if all_ok:
        print("\nAll integration tests passed successfully!")
    else:
        print("\nSome integration tests failed. Please check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
