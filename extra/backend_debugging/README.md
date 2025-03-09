# backend_debugging

### My understanding

I was given around an hour and 15 minutes to complete the assignment. In this part of the assignment, we were "not allowed to use AI", but they didn't change out the computer from the first exercise.

There was little to no explanation for this assignment, and you were just given a README.md that wasn't actually entirely accurate. 
There was a package missing called `httpx` (it could've been `httx` though, I can't remember) that was needed to run the tests, and the tests were in a file called `test_load.py`, but the README.md
had a command to run `load-test.py` to throw you off. There were accurate commands that you had to run to initialize the database and populate it with data.

The server was built using `fastapi` and was on port 8000, and you'd run `test_load.py` and it would start up the server and try to hit three AP

There were three sqlite tables, the first one being products, and the other two being something like "product_view" and "product_costs".

### Fundamental issues

When you run the tests, you'll see that the tests try to do three things:

- Make some API request to get product analytics via a cache
- Make some API request to get something else about the product
- Make some API request to do something else

In the tests, they associate each test case with a specific failure case. They output whether the requests to the endpoint
were successful and how fast they were. You'll notice that some endpoints fail completely and others are just super slow.
You'll now whether you're done or not because running the test will output speedy and successful results in the output. If I recall correctly, the failure cases were:

- An $N+1$ issue
  - Within the API endpoint implementation, it would make a request to get information about the product and then
  run two separate queries to search separately in each of the other two tables to get other information about the product.
  - You'll notice that the API takes a long time to run or fails, which is because there are too many queries. You can fix it
  by using "subqueries" to speed it up
- A caching concurrency issue
  - They had something like `asyncio.await(2)` that was causing the endpoint to be slow, and then also they had some statement
  for looking into the cache like `time.now() - thirty_days`, but since they didn't put it into a variable, they had written out separately
  exactly three times, and each time that `time.now() - thirty_days` is called, it gives a different result, which caused some
  concurrency issue. If you put that statement into a variable and just use the variable everywhere else, it should fix the issue.
  I think I remember there being some other issue that was non-obvious but you'll know if you solved it if it runs successfully or isn't slow.
- db resource issue
  - This is the easiest issue to resolve, and all you have to do is go into `database.py` to increase the size of a bunch of variables like
  `thread_pool` or `thread_count`, I can't remember exactly what the variables are called. It's super easy, just set them from 0 to 10 or more, run
  the tests, and see what happens.

There is one more issue that's not in the tests that's at the bottom of one of the database files, which is where there is a `user_id` field that doesn't
have something called a `PyDantic` `BaseModel` associated with it. This is something that you can look up quickly/resolve quickly with the FastAPI
documentation, just checking it out right now to get an idea of what `BaseModel` is for.