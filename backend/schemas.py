from pydantic import BaseModel

class CustomerData(BaseModel):
    tenure: float
    numberofaddress: float
    cashbackamount: float
    daysincelastorder: float
    ordercount: float
    satisfactionscore: float

