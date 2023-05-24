import asyncio
from random import randint

import aiohttp
import numpy as np
from sklearn.neural_network import MLPClassifier


async def get_calcs_from_web(sample5: list[int], nums: list[int]) -> tuple[list[bool], list[bool]]:

    async def get_expr(expr_url: str):
        async with session.get(expr_url) as resp:
            return await resp.text()

    async with aiohttp.ClientSession("https://api.mathjs.org") as session:
        tasks = []
        for x in sample5:
            url = f"/v4/?expr={x}%255"
            tasks.append(asyncio.ensure_future(get_expr(url)))
        for number in nums:
            url = f"/v4/?expr={number}%253"
            tasks.append(asyncio.ensure_future(get_expr(url)))

        result = await asyncio.gather(*tasks)
        result = [int(x) == 0 for x in result]
        return result[:len(sample5)], result[len(sample5):]


def main() -> None:
    print("Введите целые числа, разделённые пробелом (или оставьте пустую строку для набора случайных):")
    numbers = [int(x) for x in input(">").split()]
    if not numbers:
        numbers = [randint(1, 1_000_000) for _ in range(10)]
    print(numbers)

    x5 = list(range(10))
    y5, res3 = asyncio.run(get_calcs_from_web(x5, numbers))
    x5_test = [int(str(x)[-1]) for x in numbers]
    x5 = np.array(x5, dtype=np.int8)[:, None]
    x5_test = np.array(x5_test, dtype=np.int8)[:, None]

    # Multi-layer Perceptron neural network enters the scene...
    cool_nn = MLPClassifier(random_state=113113, solver="lbfgs", activation="tanh")
    cool_nn.fit(x5, y5)
    print("Accuracy on train dataset (should be 1.0): ", cool_nn.score(x5, y5))
    res5 = cool_nn.predict(x5_test)

    max_len = len(str(max(numbers)))
    for num, div3, div5 in zip(numbers, res3, res5):
        print(f"{num:}: ".rjust(max_len+2), "foo" if div3 else "", "bar" if div5 else "", sep="")


if __name__ == "__main__":
    main()
