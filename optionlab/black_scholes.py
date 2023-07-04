from __future__ import division
from scipy import stats,pi
from numpy import exp,round,arange,abs,argmin
from numpy.lib.scimath import log,sqrt

def getoptionprice(optype,s0,x,r,time2maturity,d1,d2):
    '''
    getoptionprice(optype,s0,x,r,time2maturity,d1,d2) -> returns the price of 
    an option (call or put) given the current stock price 's0' and the strike 
    'x', as well as the time left to maturity 'time2maturity' (in units of 
    year), 'd1' and 'd2' as defined in the Black-Scholes formula.
    '''               
    if time2maturity<=0.0:
        raise ValueError("Time left to maturity must be greater than zero!")
                         
    if optype=="call":		
        return round(s0*stats.norm.cdf(d1)-x*exp(-r*time2maturity)*stats.norm.cdf(d2),2)
    elif optype=="put":
        return round(x*exp(-r*time2maturity)*stats.norm.cdf(-d2)-s0*stats.norm.cdf(-d1),2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
        
def getimpliedvol(optype,oprice,s0,x,r,time2maturity):
    '''
    getimpliedvol(optype,oprice,s0,x,r,time2maturity) -> estimates the implied 
    volatility taking the spot price 's0', the option type (call or put), the 
    strike 'x', the time left to maturity, and the risk-free rate rate 'r' as 
    arguments.
    '''
    vol=0.001*arange(1,1001)
    d1=(log(s0/x)+(r+vol*vol/2.0)*time2maturity)/(vol*sqrt(time2maturity))
    
    if optype=="call":
        dopt=abs(round((s0*stats.norm.cdf(d1)-x*exp(-r*time2maturity)*
                        stats.norm.cdf(d1-vol*sqrt(time2maturity))),2)-oprice)
    elif optype=="put":
        dopt=abs(round((x*exp(-r*time2maturity)*stats.norm.cdf(-d1+vol*sqrt(time2maturity))-
                        s0*stats.norm.cdf(-d1)),2)-oprice)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
        
    return vol[argmin(dopt)]

def getdelta(optype,d1):
    '''
    getdelta(optype,d1) -> computes the Greek Delta for an option (call or put) 
    taking 'd1' as defined in the Black-Scholes formula as input variable. The 
    Greek Delta estimates how the option price varies as the stock price 
    increases or decreases by $1.
    '''            
    if optype=="call":
        return stats.norm.cdf(d1)
    elif optype=="put":
        return stats.norm.cdf(d1)-1.0
    else:
        raise ValueError("Option must be either 'call' or 'put'!")

def getgamma(s0,vol,time2maturity,d1):
    '''
    getgamma(s0,vol,time2maturity,d1) -> computes the Greek Gamma for options 
    taking the underlying stock price 's0', the time left before maturity, and 
    'd1' as defined in the Black-Scholes formula as input variables. The 
    Greek Gamma provides the variation of Greek Delta as stock price increases 
    or decreases by $1.
    '''        
    d1prime=1.0/sqrt(2.0*pi)*exp(-d1*d1/2.0)

    return d1prime/(s0*vol*sqrt(time2maturity))
   
def gettheta(optype,s0,x,r,vol,time2maturity,d1,d2):
    '''
    gettheta(optype,s0,x,r,vol,time2maturity,d1,d2) -> computes the Greek Theta 
    for an option (call or put) taking the underlying stock price 's0', the 
    exercise price 'x', the time left to maturity, the risk-free rate 'r', the 
    volatility 'vol', 'd1' and 'd2' as defined in the Black-Scholes formula as 
    input variables. The Greek Theta estimates the value lost per year for an 
    option as the maturity gets closer.
    '''        
    if optype=="call":
        return -((s0*vol*exp(-0.5*d1*d1))/(2.0*sqrt(2.0*pi*time2maturity))+
                 r*x*exp(-r*time2maturity)*stats.norm.cdf(d2))
    elif optype=="put":
        return -((s0*vol*exp(-0.5*d1*d1))/(2.0*sqrt(2.0*pi*time2maturity))-
                 r*x*exp(-r*time2maturity)*stats.norm.cdf(-d2))
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")

def getvega(s0,time2maturity,d1):
    '''
    getvega(s0,time2maturity,d1) -> computes the Greek Vega for options taking 
    the underlying stock price 's0', the time left before maturity, and 'd1' as 
    defined in the Black-Scholes formula as input variables. The Greek Vega 
    estimates the amount that the option price changes for every 1% change in 
    the annualized volatility of the underlying asset.
    '''
    return s0*stats.norm.pdf(d1)*sqrt(time2maturity)/100

def getcallputparity(callprice,putprice,s0,x,r,time2maturity):
    '''
    getcallputparity(callprice,putprice,s0,x,r,time2maturity) -> returns 
    the call-put parity between the options of premium 'callprice' and 
    'putprice', both with srike 'x', considering the current price 's0', 
    risk-free rate 'r' and 'time2maturity' left until the exercise.
    '''
    return callprice-putprice+x/((1+r)**time2maturity)-s0

def get_d1_d2(s0,x,r,vol,time2maturity):
    '''
    get_d1_d2(s0,x,r,vol,time2maturity) -> sets 'd1' and 'd2' for the underlying 
    stock price 's0', the exercise price 'x', the risk-free rate 'r' and the 
    volatility 'vol' with 'time2maturity' left until the option expiration.
    '''
    d1=(log(s0/x)+(r+vol*vol/2.0)*time2maturity)/(vol*sqrt(time2maturity))
    d2=d1-vol*sqrt(time2maturity)
        
    return d1,d2

def getitmprob(optype,d2):
    '''
    getitmprob(optype,d2) -> returns the estimated probability that an option 
    (either call or put) will be in-the-money at maturity, with 'd2' as defined 
    in the Black-Scholes formula provided as an input variable.
    '''        
    if optype=="call":
        return stats.norm.cdf(d2)
    elif optype=="put":
        return stats.norm.cdf(-d2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
        
def getBSinfo(stockprice,strike,rate,volatility,time2maturity):
    '''
    getBSinfo(stockprice,strike,rate,volatility,time2maturity) -> provides 
    informaton about call and put options using the Black-Scholes formula, 
    taking the stock price, the option strike, the risk-free rate, the 
    annualized volatility, an the time left to maturity as input parameters.
    '''                         
    if stockprice<=0.0:
        raise ValueError("Stock price must be greater than zero!")
    elif strike<=0.0:
        raise ValueError("Strike price must be greater than zero!")
    elif rate<=0.0:
        raise ValueError("Risk-free rate must be greater than zero!")         
    elif volatility<=0.0:
        raise ValueError("Volatility must be greater than zero!")
    elif time2maturity<=0.0:
        raise ValueError("Time left to maturity must be greater than zero!")
           
    d1,d2=get_d1_d2(stockprice,strike,rate,volatility,time2maturity)   
    callprice=getoptionprice("call",stockprice,strike,rate,time2maturity,d1,d2)
    putprice=getoptionprice("put",stockprice,strike,rate,time2maturity,d1,d2)
    calldelta=getdelta("call",d1)
    putdelta=getdelta("put",d1)
    calltheta=gettheta("call",stockprice,strike,rate,volatility,time2maturity,d1,d2)
    puttheta=gettheta("put",stockprice,strike,rate,volatility,time2maturity,d1,d2)
    gamma=getgamma(stockprice,volatility,time2maturity,d1)
    vega=getvega(stockprice,time2maturity,d1)
    callitmprob=getitmprob("call",d2)
    putitmprob=getitmprob("put",d2)
    
    return (callprice,putprice,calldelta,putdelta,calltheta,puttheta,gamma,
            vega,callitmprob,putitmprob)
