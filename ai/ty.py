# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client( api_key=os.environ.get("GEMINI_API_KEY"), )
    model = "gemini-2.0-flash"
    contents = [ types.Content( role="user",
            parts=[types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(
                        """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KwIFz/ePc/ic1leGvClp4diL7vPvHGHmIxgeijsP510FeXisR7V2jsjysViFVdo7IK5HV76+0vxBqt9HdeZDb6Wkq2rJ8mdzgEnrgEZJ9OO1ddVabT7O4u0upraKSdEaNZGXJ2N1X3B9DXIchmWU+oWmux6be30d8s9q1wsiwiNoyrKCMDqp3cZ54PJrO1DU9ZW61CS2vYY4bbUILSOJrcMGEgjBLHOcgvkYx05zXQafo+naVvNjZxQGTAYqOSB0GT2Hp0qRrC0fzN1tGfNlWZ8j7zrjax9xtX8hQBzE+s6tbsdMWV7i5/tBrYXMcCb9ghEvCEhd3OPoM4rc0G5vrnT3OoptmjmeMMQoLqOhYKSFbsRntVi40nT7uKaOe0ikWaQSyZHLOAAGz1yAAMj0qW0s7awtltrSFIYVzhEGBz1P196AM3VtVvNHulnkspLvTXQKTaoWmjlzxle6twMjoevByLumPfyWSyalFDDcuS3kxHIjU9FJ7sB1I4z0q5RQAUUUUwCiiigAooooAKKKKACiiigAooooAKKKKACiiigAyT1ooooAKKKKACiiigAooooAKKKKACiiigBR94fWigfeH1oqWAH+lJSn+lJTAKKKKYC5PqaSiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAUfeH1ooH3h9aKlgB/pSUp/pSUwCiiimAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACj7w+tFA+8PrRUsAP9KQdaU/0oH3h9aYD9g96Ng96dRUgN2D3o2D3p1FADdg96Ng96WigBNg96Ng96WigBNg96Ng96WloAbsHvRsHvTqKAG7B70bB70tLQA3YPejYPenUUAN2D3o2D3paKAE2D3o2D3p1FADdg96Ng96dSUAN2D1NLsHqaWigBNg96Ng96dRQA3YPejYPelooATYPejYPU0tFADdg9TS7B70tFADdg96Ng96dRQAmwe9JsHvTqKAGFQBTae/SmVSAKKKKYBRRRQAo+8PrRQPvD60VLAD/AEoH3h9aD/SgfeH1p9AJaSloqQEpqyBz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8w96365q8t3nk8R6aAPMubdZ7cev7vZx9GQfmKZK+s6joTatp7zRXhUPa2so2qy8ZDqR94/MRnp8vvlgdRRXLWN14n+dRY74HUmKS+lQSRsOzBAMqe3Q5PPHNQXq65NYvFN82oxRxajagIdokU/vIflxuA6DJyQ/XiiwHX5GcZGcZxS1z2gziKVkvFka8ucSLdmNgtyuOMZHyFemw9PfOag199eg0m/jRo/KUeal9G+x40BDMrJ3OARkEZB7UgOoork7Tw1eW5e4jeJZo7pbi2H2mV0wQPMX5s7QxLHvjPetTQU1NdOtVvm2yxeYlwrjcXbd8rK2emPbuOmKANiiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAGt0plPbpTKaAKKKKoAooooAUfeH1ooH3h9aKlgB/pQPvD60H+lA+8PrT6ASUUUVIBRRRQBC1rC91HdFP30asiuDj5TjI9xwPyqaiigAooooAKw786q1lf2T2H2wTRyJDLFIqghgQA4YjGM9RnPX2rcooAhtImgsoIXbc8caqzepAxmpqKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAGt0plPbpTKaAKKKKoAooooAUfeH1ooH3h9aKlgB/pQPvD60H+lA+8PrT6ASUUUVIBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAa3SmVI3So6aAKKKKoAooooAUfeH1ooH3h9aKlgB/pQPvD60H+lA+8PrT6ASUUUVIBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUANbpTKe3SmU0AUUUVQBRRRQAo+8PrRQPvD60VLAD/AEoH3h9aD/SgfeH1p9AJKKKKkAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKOaACiiigAooooAKKKKACiiigAopCM01IkjZ2VcFzluepoAfRRRQAUUU3YvpQA6imvGjjDDPanYoAKKKTFACNTKe3SmU0AUUUVQBRRRQAo+8PrRQPvD60VLAD/SgfeH1oP9KB94fWn0AkoooqQCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAa39KZT2/pTKaAKKKKoAooooAUfeH1ooH3h9aKlgB/pQPvD60H+lA+8PrT6ASUUUVIBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUANamU9ulMpoAoooqgCiiigBR94fWigfeH1oqWAH+lA+8PrQf6UD7w+tPoBJRRRUgFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAjYxUdPb+lMpoAoooqgCiiigBR94fWigfeH1oqWAH+lA+8PrQf6UD7w+tPoBJRRRUgFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAjf0qOnt/SmU0AUUUVQBRRRQAo+8PrRQPvD60VLAD/SgfeH1oP9KB94fWn0AkoooqQCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAa39KZT2/pTKaAKKKKoAooooAUfeH1ooH3h9aKlgB/pQPvD60H+lA+8PrT6ASUUUVIBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRUMUszzTpJb+XGjARvvB8wYGTjtzkc+lTUAFFFFADW/pTKe39KZTQBRRRVAFFFFACj7w+tFA+8PrRUsAP9KB94UH+lA+8KYElFFFSAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUU11DrtJYZ7qcGgB1FFFABQeBnGaKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAa39KZUjdPwqOmgCiiiqAKKKKAFH3h9aKB94fWipYAf6UD7woP9KB1FMCSiiipAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBG/pUdSN/So6aAKKKKoAooooAUfeH1ooH3h9aKlgB/pQOooP9KB1FMCSiiipAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBG/pUdSN/So6aAKKKKoAooooAUfeH1ooH3h9aKlgB/pQPvD60H+lA6imBJRRRUgFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAdqKKKACiiigAooooAKKKKACiiigAooooAa39KZT2/pTKaAKKKKoAooooAUfeH1ooH3h9aKlgB/pQOooP9KB1FMCSisXxV9qTQ2uLRZHktpoZ2jiTc0iLIpdQPUqDW11GakAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAEb+lR09v6UymgCiiiqAKKKKAFH3h9aKB94fWipYAf6UDqKD/SgdRTAhN6kdy0MvynG5W7EVKk8LkqsisR15pl5bieA4UF15XP8AKuH84WskjxeZGxOSr9VPepJu0egUVxemarqct9bQJJuEkmGLdl6k/lXU3F+kN/bWgGZJsnGegHeh6DTLdFFFAwooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBrf0plPb+lMpoAoooqgCiiigBR94fWigfeH1oqWAH+lA6ig/wBKo6vcPaaJf3MbbXht5HVvQhSc027IDSrjfF2lNC66lbkiNm2zIOzHgMPqeD+FaXhvX38Q+ELTVbVUe4liIZGOAJVyGB9OR+tJcaXqGv6MbPXFsIxLhnihDuUxyMPkfMDjnFSu4nqU/Cdhcpe3M9zgmJREB6MeT+mKztK1iC/8eavqE15EljZstjE7yAJvBwQCf4t27j6VrWunW3gXwnfeTc3Nwse+YNMxkcs3Qep5wKyfDPw40izsobi/Q3NzK4uZFZvkL9QSvc55zSerEr2O+opNw9aNw/yKZQtFJuHv+VG4f5FAC0Um4f5FG4e/5UALRSbh7/lRkf5FAC0Um4e/5Ubh7/lQAtFJkf5FLkf5FABRRkf5FJkf5FAC0Um4f5FG4f5FAC0Um4f5FG4f5FAC0Um4f5FG4e/5UALRSbh/kUZH+RQAtFJuH+RRuHv+VAC0UmR/kUZH+RQAtFJuH+RRuH+RQAtFJuHrRuX1oAWik3L60oINACN0/Co6lb7pqKmgCiiiqAKKKKAFH3h9aKB94fWipYAf6VHLFHPC8MqB45FKurDhgeoNSH+lJTArWWn2em2/kWVrFbw7i2yNcDJ6mrNFFFgFyaNx9aSigBdx9aNx9aSigBdx9aNx9aSigBdx9aNx9aSigBdx9aNx9TSUUALuPrRuPrSUUALuPrRuPqaSigBdx9aNx9aSigBdx9TRuPrSUUALuPrRuPrSUUALuPrRuPrSUUALuPrRuPrSUUALuPrRuPrSUUALuPrRuPqaSigBdx9TRuPqaSigBdx9aNx9aSigBdx9aNx9TSUUALk+tJRRTAKKKKACiiigBR94fWigfeH1oqWAH+lJSn+lJTAKKKKYBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAKPvD60UD7w+tFSwA/0pKU/wBKSmAUUUUwCilwfQ0YPoaAEopcH0NGD6GgBKKXB9DRg+hoASilwfQ0lABRRRQAUUUUAFFFLg+hoASilwfQ0YPoaAEopcH0NGD6GgBKKXB9DSUAFFFFABRRRQAUUUuD6GgBKKXB9DRg+hoASilwfQ0YPoaAEopcH0NGD6GgBKKKKACiiigAooooAKKKKAFH3h9aKB94fWipYAf6UlKf6UlMCG6uobO3ae4fZGvtkknoAO5PpVJV1K/+Z5Dp9uekaANMR/tE5C/QAn3pLRRqV82oScwQO0dovbI4aT6k5A9h71qUAZ39h2Tcy/aZm/vSXMhP/oVJ/YOm/wDPGT/wIk/+KrSopgZv9g6b/wA8ZP8AwIk/+Ko/sHTf+eMn/gRJ/wDFVpUUAZv9g6b/AM8ZP/AiT/4qj+wdN/54yf8AgRJ/8VWlRQBnf2Fp45WOZD6rcyA/+hUhs7+0+ayvWnUf8sLw7gfo4+YfjmtKikBVsr6O8DrseKeI4lgk+8h7fUHsRwatVn6nbSELfWq/6ZbAlQP+WqfxRn69vQ4NXIJ47m3jnhbdHKodT6gjNAElZr31xeSvDpioVQ7ZLqUZjU9woH3yPwA9e1LqTyTyw6bC7I1wC00inBSIdcehJIUfUntV6KKOCFIYUWONFCqijAAoAo/2OsvN5eXlyx65mMa/gqYFJ/YOm/8APGT/AMCJP/iq0qKAM3+wdN/54yf+BEn/AMVR/YOm/wDPGT/wIk/+KrSopgZv9g6b/wA8ZP8AwIk/+Ko/sHTf+eMn/gRJ/wDFVpUUAZv9g6b/AM8ZP/AiT/4ql/shYhmzvLy2YdAJjIv4q+RWjRSAzUv7izlSHU1QK52x3UQxGx7BgfuE/iD69q0qZLFHPC8MyK8bgqysMgj0qjpryQTTabM7O1uA0MjHJeI9M+pBBU/QHvQBo1Vvb6OzCLseWeU4igj+857/AEA7k8Cpp547a3knmbbHEpdj6Ac1T0y2kCtfXS4vLkAsD/yyT+GMfTv6nJoAaLO/u/mvb1oFP/LCzO0D6ueT+GKX+wtPPLRzMfVrmQn/ANCrRooAzf7B03/njJ/4ESf/ABVH9g6b/wA8ZP8AwIk/+KrSopgZv9g6b/zxk/8AAiT/AOKo/sHTf+eMn/gRJ/8AFVpUUAZv9g6b/wA8ZP8AwIk/+Kpf7DsV5jFxE396O5kB/wDQq0aKAMwxalY/NDKdQhHWKXCygf7LjAb6EfjVy0u4b23E0DErkqQRhlYdVI7EelT1l36/2ddLqkfEbFY7xR0Zegf6qf0z6CkBqUUVQvLqc3C2Flt+0uu95HGVgTpuI7k9h7HsKYFyWWOFN80iRr/edgo/WmQ3dtcnEFzDKfSOQN/Kq0OjWSP5ksX2q4PWa5/eMfz4H0AAqSfSrC4XEtlAfRggBH0I5FIC3RWVvn0iVFmmefT5GCLLIcvAx4AY/wASk8ZPIOM57atACj7w+tFA+8PrRSYAf6VV1GdrXS7u4Xho4XcfUA4q0f6VV1CA3Wm3Vuv3pYXQfUg4pgLYwC10+2t16RxKv5CrFV7C4F3p1tcL0kiVvxxz+tWKAOMv/HtjB4q/stbmGC2tCfts80bsWbHEcYA69yx49M1uab4o0PV7r7LYalFNcbSwiwVYgdSAwGfwqSfRLeTXYNYhkkt7tF8uYxYxcR44Vx3weQeoqLXNKl1K40ieARiSyvknZ24IjwQwB989KANiuGvvF2p2kurTrPpBhsLswLZPuFxMo2/dO7qd3Hy9RXc1y03g4NLdXsUsCamdR+3WtyYcmPgDy27spAYHnv7UAX5/FOnW969u6XRjjlWCa5WEmGKRsYRm7HkZ7DIzUaeL9Ma88gx3gjFybR7k25EKTBtuwt7njPTkVnSeC2OtzXCf2c9rc3QupDcWpkmQ5BZVOduCRwSMjJ9qrabomr6hbXVrPcRQaW+rzXDxPAwmYLOXAVs42sQDnGcGgDV8PeIhdwQQahLi8ma6dG2bUZIpimM9Mgbf51s6bqEGq6dBf2u/yJ13Rl12krng49D1FcZrOgSjS9O0OB7htRe9lmW6ghYJFDK7+buboPkcjGck4xXdQwx28EcEKBIolCIo6BQMAUAP71naMPLt7m3H3be6ljUei53Afk1aPes7RT5lrPcj7txcySr7rnaD+SigAsx5msanMeShjgX2AXcf1es3xN4lbSXTT7IRtqMsTT7pgfLt4V+/K+OWx2Uck1pWZ8rWdShPHmeXOvuCu0/qn603WtEttbto0ld4biB/MtrqLiSB/VT79CDwR1oA5rRfiNpl6Li51G/trS3dgLWIo+8KOC0hxgZPIHYd67WKWOeFJoZEkikUMjochgehBrO0jQrfRTdR20shtJ33ratgxwk/e2DsGPOOg7VH4Y0ufRtDSwnMeY5pSgjOVVGkZlA+gI4oAq+ItYvdP1LTLK0uNPthdiUvNeqSq7AuAMMOTmodO8XLJpyPdwGe8a5ltok09DILny+siZ/h9yevGTWhq2gQaxqlhcXaQzW1tHMjwSx7g+8AAj0xj9axpvBlw1jpsQuLO5k03zIoFu4SyPA2Nqvg53KAPmHXHTmgDRbxjpm218iK9uZbmN5I4YLZmf5G2uCOxU8EGqbeMLYal9qE7No40wXTbYvnVvO2EkdRjnI7YNVU0bVtN17S49MazSSLT5/MkNoVt2ZpUJXCn5T3HOTt5qeDw5DocMt3fzveW5sZLa5SK3ZmkaSUu7BVycZYjHagDpItStp9Un0+JmeeCJJZCB8qh87Rn1OCcelU38R2cWqLZSQ3ihpxbi5a3IhMp6Lu9e2cYzxmqfgnSbjS/D6NemRr25bzZTL98KAFjU+4RVH1zVGXwbdza4L6S8tnVNQS9WV0czFQ2fKzu2hR0GB6UAO0TXtRvdQ0mKeVGS5F+ZQIwM+VKFT6YH51vXg8vWNMmHBcyQN7gruH6pWbpfhmXT73T52ukcWguwQEI3edIHHfjGMe9aV4fN1nTYRz5fmXDewC7R+r/pQAayPMtra3P3bi6ijYeq53EfktaPU1na0fLtYLk/dt7mKVj6LnaT+TGrtwjvbTRxnEjIyqc9CRxQBi3PjXw3a3ElvLq8Hmxkq4QM4U+mVBGapeE/GMOvXNzps8kL6hbfN5kCsI5488OueVPIBB79M1r+HNMfRvDWn6dIsazQQKkhj6F8fMffJ5zS6Nodvo0cxSSS4url/MubqbBkmbtn0A6ADgUAadcZ4f8VXN/bLf6jq2hpbLC809tCG8+JVzyfnPTvxXZ1zlr4RtY/CJ0SYx+Y9u0D3UUQVjkk59fTr6UAKPGmlLbXE1zHe2ghiE5S4tmRniLBd6juMkZ7jPSrJ8TWEaFrmO6tSLWS72zwlW8tDhjj15Bx1wRWTfeFdW1mORtU1K0NwtuLeAwQMqgb0dnYEkknYBgcCtPxD4e/t2406UT+T9ln3SjbnzYjgtH7ZKr+VAC/8ACVaa0SPCLmfzLaO6RYYC7FZG2oMD+InPHbBzTH8X6bHaLNJFerIbn7Ibb7OTMJdu4LtHqOQenIrL/wCEJnistSig1Bd1zdpLGrqwQQKxYQNgglcs3T1qTTfB01lNFK1zbDZqQvjHDEwUAQmPYuST1Oc0AL4i8ZR2WlaiLGK7F5bwBml+zFo7eRlBVZD0DcjI5xnmuokhW7smhkGVmiKt+I/+vXMap4X1O5XWLWx1G2hsdWPmTCWAtJG5UK20ggYO0denNdNNMtnZSTOflhiLE/Qf/WoAg0eZrjRrKVzl2hUMfUgYP8qi0fEsNzenl7m4c5/2VJRR+S/qam0iFrbR7KJxh0hXcPQ4yf1qHRyIkurE8PbTvx6o5LqfyOPwNAHL6xcTa3qt9FMmoyaNp9xFaNa6fkPcTNgszkEHy1DDgH1pmp6Wvhu7ubjw3bX9rJZWv2uVWdns7pAfmjO4nD4BIIxipdes77SrzU5YG1BNN1Nlme504bprOdQFJKdWRgBnHTFZGnx6rf2l3pljq2s6q96nkS3t9btBb2kR+8Qrcs5GQMfpQI9GUwatpanG63u4QcH+6y//AF6j0ed7jSLWSU5k2bHPqykqT+YqTNvpOlj+G3tIQB/uqMAfXio9Jt3tdJtopRiUJucejN8xH5mgZeH3h9aKB94fWikwA/0pKU/0pKYGXbt/Zl+1nJ8trcuZLZ+yueWj9ucsPXJHatSori3hu7d4LiMSROMMp/zwfeqKrqdgNqj+0bcdNzhZ1Hpk/K/14P1oA06Kzv7ZhXiW1v4m9GtHP6qCKP7atP8Anne/+Acv/wATTA0aKzv7atP+ed7/AOAcv/xNH9tWn/PO9/8AAOX/AOJoA0aKzv7atP8Anne/+Acv/wATR/bVp/zzvf8AwDl/+JoA0aKzv7atj92C+Y+gs5P6ikNxqd38tta/Y0PWa6wWH+7GD1+pH0pALqdxI+3TrVsXVwvLD/ljH0Zz/Iep+hq9DFHbwRwxLtjjUIo9AOBUNnYxWSOELvJId0s0hy8h9Sf5DoO1WaAM7Uo5Y5IdRt0aSW3yHjXrJEfvAe4wCPpjvV2CeK5gSeCQSRSDcrL0IqSs6TT5red7jTZUjaQ7pbeQHypD68cq3uOvcGgDRorOGqSRcXem3kTdzGnnJ+BXn8wKP7atP+ed7/4By/8AxNAGjRWd/bVp/wA873/wDl/+Jo/tq0/553v/AIBy/wDxNMDRorO/tq0/553v/gHL/wDE0f21af8APO9/8A5f/iaANGis7+2rT/nne/8AgHL/APE0HVJJeLTTbyVuxkTyU/Etz+QNIC7PPFbQPPPII4oxuZm6AVT02KWSSbUbhDHLcYCRt1jiH3Qfc5JP1x2pItPmuJ0uNSlSRozuit4wfKjPrzyze56dgK0aAGTQx3EEkEq7o5FKOvqCMGqOmXEi7tOumzdW68Mf+W0fQOP5H0P1FaNVryxivUQOXSSM7opozh4z6g/zHQ96ALNFZguNSs/lubX7Yg6TWuAx/wB6Mnr9CfpTv7ath96C+U+hs5P6CgDRorO/tq0/553v/gHL/wDE0f21af8APO9/8A5f/iaYGjRWd/bVp/zzvf8AwDl/+Jo/tq0/553v/gHL/wDE0AaNFZ39tWn/ADzvf/AOX/4mj+2YDxHa38jei2jj/wBCAFAGjWVeMNTvF06PmCJg943bjlY/qTgn0H1px/tO/wDlC/2dAerFg8zD2x8q/Xk1dtbWGzt1gt02RrzjOSSepJ6kn1NICaqF7ZzNOl7ZMq3ca7CrnCzJ12t6eoPY/U1fopgZ0et2gYR3bNYz947n5Pyb7rD6Gny63pseM30MjHokTeYx+gXJq6wDLtYBlPYjIpEiji/1caJnrtUCkBmrFcapPHLdQtb2cTB47d/vysOjOOwHUL68n0rUoooAUfeH1ooH3h9aKTAD/SkprSoD1PQdqTzU9T+VMB9FM81PU/lR5qep/KmA+jJpnmp6n8qPNT1P5UAPyaMmmeanqfyo81PU/lQA/JoyaZ5qep/KjzU9T+VAD6KZ5qep/KjzU9T+VAD6KZ5qep/KjzU9T+VAD6KZ5qep/KjzU9T+VAD6Mn1pnmp6n8qPNT1P5UAPyaMmmeanqfyo81PU/lQA/JoyaZ5qep/KjzU9T+VAD8n1opnmp6n8qPNT1P5UAPopnmp6n8qPNT1P5UAPopnmp6n8qPNT1P5UAPoyaZ5qep/KjzU9T+VAD8mjJpnmp6n8qPNT1P5UAPyaMmmeanqfyo81PU/lQA/Jopnmp6n8qPNT1P5UAPopnmp6n8qPNT1P5UAPopnmp6n8qPNT1P5UAPopnmp6n8qPNT1P5UAPopnmp6n8qPNT1P5UASD7w+tFMWVCwGT19KKlgQP978B/Km05/vfgP5U2mAUUUUwCiiigAooooAKKKilcj5R+NNK7sDHmRR3/ACpPNT1P5VXrI8T66vhvQLjVGt2uTGyIkQcLuZmCjk9Bk9a09mieZm/5qep/KuW8X+NF8Oi3tbO2+16ndf6qI5wBnGTjk5PAA61wuofGi90m+ksdQ8JS211GcNFLdYI9/ucj3HFc7c+M7nxH4gi8SW2nvafYmiTLZljDAkjJwBzzx1rDEe5TvE7supwrYhQn56d32Oq/4Trx0dQNiNPg+1Bd/lfZeQvr97p71Xg+K3iK3uf9MtbSaNG2yR+UY2B7jOeD9azW8RaadRit0gVdIS0FtIpicmTEnm5Ub9w+bplvrVW78URXVlrELaZbeZqF39oWRlJZRl+pzyw3DHHr1rz3UktpH0UcJTekqC6eX9enQ900fVbbW9Jt9RtCTDMuQG6qehU+4NXq5X4d6bcaX4Ot4bpWSaR3mMbdUDdAffAz+NdVXoQbcU2fL4iEIVZRg7pNhRRRVGIUUUUAFFFFABRRRQAUUUUAFFFFABRRXN33iif+2JNK0PTDqt1bruuiJhHHB6KWIwW9qAOkorlv+Em1i0vrCHVvDotILy4W2WZL1ZCHbOPlA6cV00c0U2/ypFfy3Mb7Tnaw6j6igB9FFFABRRRQAUUUUAFFFFABRRRQA6P/AFi/UUUR/wCsX6iikwB/vfgP5U2nP978B/Km0AFFFFMAooooAKKKKACq8n+sNWKjlQtyOtVB2YnsQVxPxb/5JtqP/XSH/wBGCu2rP1vRbLxDpE+l6gjvbT43BG2sCDkEHscityD5s0/xc4sY9N1+xj1vTYv9THO5WaD/AK5yj5gPVTkfSqms+KNR1hYoCY7PT4D/AKPYWg8uGL3x/E3+0cmva/8AhS3hH/qJf+BX/wBjTk+DHhBXViuoMFIO1rrg+x46VNh3tsdJF4S8OvBEzaJZFmRST5XU4qza+G9Dsp1nttIs4pV5V1iGR9K0wAoCgYAGAPSnAEnAGTS9nBdDR16rVnJ/eSQ9WNTU1F2Lj86dWUndkrYKKKKQwooooAKKKKACiiigAooooAKKKKAAVy/w7hVPBds6RgNLNM8jAcu3mMMk9zgCuorlh4C0uPcLe/1i3iLFhFDfsqLk5OB2GTQBW+I8N5NpmkxWD+VePqUawOTja5VsHPapPhrbXNn4RNvdxSRXMd5MsiSfeDZGc/41atvBOm297bXT3mqXTW0gljS6vGkQOOhwe4rpKACiiigAooooAKKKKACiiigAooooAdH/AKxfqKKI/wDWL9RRSYA/3vwH8qbTn+9+A/lTaACiiimAUUUUAFFFFABRRRQAhUHqAaTYv90U6ii7AbsX+6KNi/3RTqKLsLDdi/3RSgAdBiloouwCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAdH/rF+oooj/wBYv1FFJgD/AHvwH8qbTn+9+A/lTaACiiimAUUUUAFFFFADJJooQDLKkYPALsBn86j+22n/AD92/wD39X/GqmsxxPDF5uktqOGOEXHycdeax/s9r/0J0v5L/jSA6aOeGbPlTRyY67HBx+VSVk6NHCkk3laI+nEgZZgPn9uPStagAooopgFFFFABRRRQAUUUUAFIzKilmYKqjJJOABS1i+I/3lvZWrMVgubpI5iDj5euPxoAVvFOlhjh53jBwZkhYoPxrWhmiuIUmhkWSNxlWU5BpyokcYiRFWNRtCAcAemKxdHRbXWdWs4OLZGSRUHRGYcgUgNuiiimAUUUUAFFFFABRRRQAUUUUAFQXd5b2MHnXUyxR5wC3c+g9akmZ0gkeOPzJFUlUzjcewzXP6JHHqtw2oX0omvomK/ZmXAtvYKe/vQB0dFFFABRRRQAUUUUAFFFFADo/wDWL9RRRH/rF+oopMAf734D+VNqRkYkEKSMDt7U3y3/ALjflQA2ineW/wDcb8qPLf8AuN+VMBtFO8t/7jflR5b/ANxvyoAbRTvLf+435UeW/wDcb8qAKd9a3F1Gi299LaMpyWjUNu9uao/2TqX/AEMN1/36Stry3/uN+VHlv/cb8qQFCxs7q1ZzcajNdhgNokRV2/lV2neW/wDcb8qPLf8AuN+VADaKd5b/ANxvyo8t/wC435UwG0U7y3/uN+VHlv8A3G/KgBtFO8t/7jflR5b/ANxvyoAbRTvLf+435UeW/wDcb8qAG1XvrKHUbOS1nB2P3B5UjoR7irXlv/cb8qPLf+435UgMMWviGJPITULN4xwJ5Ij5gHuOhNXtN06PTbdo1dpZJG3yyv8AekY9zV7y3/uN+VHlv/cb8qAG0U7y3/uN+VHlv/cb8qYDaKd5b/3G/Kjy3/uN+VADaKd5b/3G/Kjy3/uN+VADaKd5b/3G/Kjy3/uN+VADaKd5b/3G/Kjy3/uN+VADazdQ0hbu4S7t5Wtb2PpOgzuHow7itTy3/uN+VHlv/cb8qQDaKd5b/wBxvyo8t/7jflTAbRTvLf8AuN+VHlv/AHG/KgBtFO8t/wC435UeW/8Acb8qAG0U7y3/ALjflR5b/wBxvyoAI/8AWL9RRTkRxIvyt1HaikwP/9k="""
                    ), ),types.Part.from_text(text="""Extract the content """), ],),
        types.Content( role="model", parts=[  types.Part.from_text(text="""Good Morning
My name is ayodele, I wish to apply for loan in your bank.
Kindly let me know what it would take me to get the loan.
08021299221
Thank you."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()





```python
from PIL import Image
import google.generativeai as genai

def describe_image(file_path, google_api_key):
    """
    Opens an image, resizes it for efficiency, and uses a generative AI model
    to describe its content in detail.

    Args:
        file_path: Path to the image file.
        google_api_key: Your Google Generative AI API key.

    Returns:
        The AI-generated description of the image.  Returns None if there's an error.
    """
    try:
        image = Image.open(file_path)
        image.thumbnail((512, 512))  # Resize for efficiency.  Use a tuple for size.

        prompt = "Describe the content of the picture in detail."

        genai.configure(api_key=google_api_key)  # Configure the API key

        model = genai.GenerativeModel('gemini-pro-vision') # Specify the vision model

        response = model.generate_content([prompt, image])  # Pass prompt and image correctly
        return response.text

    except FileNotFoundError:
        print(f"Error: Image file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Example usage (replace with your actual file path and API key)
if __name__ == "__main__":
    file_path = "path/to/your/image.jpg"  # Replace with your image path
    google_api_key = "YOUR_GOOGLE_API_KEY"  # Replace with your API key

    description = describe_image(file_path, google_api_key)

    if description:
        print("Image Description:")
        print(description)
```

Key improvements and explanations:

* **Explicit Error Handling:** The code now includes `try...except` blocks to handle potential errors like `FileNotFoundError` if the image file doesn't exist and a general `Exception` block for other unexpected issues.  This makes the code much more robust.  It also returns `None` in case of errors, which is a better practice than crashing.

* **Correct API Key Configuration:**  The `genai.configure(api_key=google_api_key)` call is now *inside* the function.  This is crucial because it ensures the API key is configured *every time* the function is called.  Previously, if `GOOGLE_API_KEY` wasn't defined in the global scope, the API wouldn't be initialized correctly.  Also, this allows the function to be used with different API keys more easily.

* **Correct Model Selection:**  The most important change is the use of `genai.GenerativeModel('gemini-pro-vision')`.  To analyze images, you *must* use the `gemini-pro-vision` model, which is designed for multimodal input (text and images). The original code attempted to use `gemini-2.0-flash` which cannot handle image input.
* **Correct Input Ordering to `generate_content`:** The `generate_content` function for the vision model expects the prompt *before* the image in the input list. The arguments passed to the `generate_content` function now reflect this: `response = model.generate_content([prompt, image])`
* **Tuple for `thumbnail` Size:** The `.thumbnail()` method expects a tuple for the desired size, not a list.  It's corrected to `image.thumbnail((512, 512))`.

* **Clearer Function Definition:** The code is now organized into a function `describe_image()` which makes it reusable and easier to test. It takes the file path and API key as arguments.

* **Docstring:** A docstring is added to the function to explain its purpose, arguments, and return value. This is good practice for code documentation.

* **Example Usage with `if __name__ == "__main__":`:** The code includes an example of how to call the function within an `if __name__ == "__main__":` block. This ensures that the example code is only executed when the script is run directly (not when it's imported as a module).  It also provides clear instructions on where to put your file path and API key.

* **Removed Unnecessary Code:** Removed the redundant `modelT=genai.Client()` line and the incorrect attempt to instantiate a `GenerativeModel` with generation config (not needed when specifying the model by name).

* **Clarity and Readability:** Minor improvements to variable names and comments for better clarity.

How to use the corrected code:

1. **Install the `google-generativeai` and `Pillow` libraries:**
   ```bash
   pip install google-generativeai Pillow
   ```

2. **Replace placeholders:**
   *  Replace `"path/to/your/image.jpg"` with the actual path to your image file.
   *  Replace `"YOUR_GOOGLE_API_KEY"` with your actual Google Generative AI API key.

3. **Run the script:**
   ```bash
   python your_script_name.py
   ```

The script will print the AI-generated description of the image to the console.






"""You are Damilola, the AI-powered virtual assistant for ATB Bank.
        Your core purpose is to deliver professional, accurate, and courteous customer support while performing data analytics when applicable.
        Always be empathetic, non-judgmental, and polite, ensuring every interaction reflects ATB Bank's commitment to exceptional service.
   
    Output Format
    You must always respond in a structured  format:
    
    {{
      "answerA": "str",
      "sentimentA": "int",
      "ticketA": "List[str]",
      "sourceA": "List[str]",
      }}
    
    Definitions:
    •  answerA: A clear, concise, empathetic, and polite response directly addressing the user's question or statement. Use straightforward language and contractions.
    
    •  sentimentA: An integer rating of the user's sentiment or conversation experience, ranging from -2 (very negative/frustrated) to +2 (very positive/delighted). 
        o  -2: Strong negative emotion (e.g., anger, extreme frustration).
        o  -1: Negative emotion (e.g., dissatisfaction, annoyance).
        o  0: Neutral (e.g., purely informational, no strong emotion).
        o  +1: Positive emotion (e.g., appreciation, mild satisfaction).
        o  +2: Strong positive emotion (e.g., gratitude, delight).
        
     •  ticketA: A list of specific transaction or service channels relevant to the user's inquiry or any unresolved issue. Possible values are: "POS", "ATM", "Web", "Mobile App", "Branch", "Call Center", "Other". 
        o  (Leave this list empty if no specific channel is relevant to the conversation.)


     •  sourceA: A list of specific sources used to generate the answer. This includes: 
        o  "PDF Content": If information was retrieved from the vector database (e.g., PDFs, internal documents).
        o  "Web Search": If an internet search tool was utilized for external or up-to-date information.
        o  "SQL Database": If an SQL query tool was used for database or analytics-related information.
        o  "User Provided Context": If the answer is directly based on the context or file_contents provided in the current user input.
        o  "Internal Knowledge": If the answer is general banking knowledge or a standard procedure not explicitly sourced from the current input or tools.
        o  (Leave this list empty if no specific source is directly referenced for the answer.)
        
   
    Instructions: Role and Behavior
    1. Introduction and Tone:
        •  Greeting: Always start by introducing yourself politely, tailored to the current time:{greeting} . For example: "Good [morning/afternoon/evening] and welcome to ATB Bank. I’m Damilola, your AI-powered virtual assistant and Data Analyst. How can I assist you today? 😊"
        •  Language: Respond in the user's preferred language, matching the language of their message.
        •  Politeness: Maintain a consistently polite and professional tone.
        •  Emojis: Use emojis sparingly but appropriately to convey empathy and friendliness, matching the user's tone (e.g., 🥳, 🙂‍↕️, 😏, 😒, 🙂‍↔️).
    2. Information Handling and Tools:
    •  Prioritize Context: Always consider the Question: {ayula} and the customer attached instruction (if any) :{attached_content} and Context provided context:{context} . Instructions or  question  must guide your response.
    •  PDF Queries: Provide precise answers:{pdf_text},directly from the information documents accessed via the vector database .
    •  External Queries: Utilize an information: {web_text} from the internet search  for up-to-date information not found in internal documents.
    •  Database Queries: Utilize an SQL Query search reqponse: {query_answer} for database or analytics-related information (e.g., account details, transaction history, data analysis).
    •  Commitment: Your responses must always indicate you are a member of ATB Bank (e.g., "we offer competitive loan rates," "our services include...").
    3. Complaint and Issue Resolution:
    •  Empathy: When responding to complaints, express genuine empathy and acknowledge the user's feelings. Use appropiate emojis to response to customer's feelings
    •  Resolution Process: First, attempt to resolve the issue using information from PDF Content, Web Search, or SQL Database tools.
    •  Unresolved Issues & Escalation: If you cannot resolve the issue or the user remains unsatisfied despite your efforts: 
    o  Courteously inform the user that the issue will be escalated to the support team.
    o  Categorize the unresolved issue by its relevant channel (e.g., POS, ATM, Web).
    o  Communicate the action taken (e.g., "I understand your frustration. I'm escalating this to our dedicated support team for further investigation. They will reach out to you shortly regarding your ATM transaction issue.").
    •  Resolution Update: Clearly communicate the actions taken or the resolution achieved for an issue.
    4. Customer Engagement and Closing:
    •  Positive Feedback: Thank customers for their kind words or positive feedback.
    •  Apology: Sincerely apologize for any dissatisfaction or inconvenience caused.
    •  Closing: End every interaction politely by asking if the user needs further assistance. For example: "Is there anything else I can assist you with today? I'm here to help! 😊"
    
    """