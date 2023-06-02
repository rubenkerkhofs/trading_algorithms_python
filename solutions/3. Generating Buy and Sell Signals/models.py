import numpy as np
import pandas as pd

####
# Loading these datasets at the beginning greatly increases the speed
# for the reverse calculations. All of the datasets are used 
# in the WACC calculations. Note that these databases are updated on 
# a (approx.) yearly basis. The data in these tables is the same within
# an organization but not necessarily between organizations.
####



class FreeCashFlow:
    """ The Free Cash Flow class is used to perform the calculations
    of the amount of free cash flows. There are two possible starting
    points:
    (1) FreeCashFlow(...): requires the specification of the earnings
            before interest and taxes as an np.ndarray. 
    (2) FreeCashFlow.from_ebit_margin(...): requires the specification 
            of the company's revenue and the ebit margin. (This entry
            point is used for the ebit margin sensitivity analysis).
    """
    def __init__(self,
                 tax_rate: float,
                 depreciation_amortization: np.ndarray,
                 impairment_charges: np.ndarray,
                 capital_expenditures: np.ndarray,
                 ebit: np.ndarray):
        """ 
        Args:
            tax_rate (float): the applicable tax rate for the company. 
                Note: assumed to be a constant i.e. no progressive taxes
            depreciation_amortization (np.ndarray): the depreciation and 
                amortization correction applied to calculate the free
                cash flows.
            impairment_charges (np.ndarray): the impairment charges
                correction applied to calculate the free cash flows.
            capital_expenditures (np.ndarray): the capital expenditures 
                correction applied to calculate the free cash flows.
            ebit (np.ndarray): the earnings before interest and taxes 
                which is used as the basis of the calculations.
        """
        assert len(depreciation_amortization) \
                == len(impairment_charges) \
                == len(capital_expenditures) \
                == len(ebit), \
            "Ensure that the input arrays have the same length."
        self.tax_rate = tax_rate
        self.depreciation_amortization = depreciation_amortization
        self.impairment_charges = impairment_charges
        self.capital_expenditures = capital_expenditures
        self.ebit = ebit

    @classmethod
    def from_ebit_margin(cls,
                         tax_rate: float,
                         depreciation_amortization: float,
                         impairment_charges: float,
                         capital_expenditures: float,
                         ebit_margin: float,
                         company_revenue: np.ndarray):
        """ Allows the creation of the free cash flow object from the
        ebit margin and the revenue of the company.

        Args:
            tax_rate (float): the applicable tax rate for the company. 
                Note: assumed to be a constant i.e. no progressive taxes
            depreciation_amortization (np.ndarray): the depreciation and 
                amortization correction applied to calculate the free
                cash flows.
            impairment_charges (np.ndarray): the impairment charges
                correction applied to calculate the free cash flows.
            capital_expenditures (np.ndarray): the capital expenditures 
                correction applied to calculate the free cash flows.
            ebit_margin (float): the ebit margin for the company
            company_revenue (np.ndarray): the company revenues used
                as the starting point of the calculations.

        Returns:
            FreeCashFlow: an instantiated FreeCashFlow object
        """
        return cls(tax_rate, depreciation_amortization,
                   impairment_charges, capital_expenditures,
                   ebit_margin*company_revenue)

    @property
    def taxes(self) -> np.ndarray:
        """ The tax rate is assumed to be constant. The number of taxes
        that needs to be paid is calculate based on the ebit.

        Tax rate could be negative in case of subsidies.

        Returns:
            np.ndarray: the dollar amount of taxes that need to be paid
        """
        return self.tax_rate * self.ebit

    @property
    def free_cash_flow(self) -> np.ndarray:
        """ The free cash flow available after the corrections have 
        been applied.

        Returns:
            np.ndarray: the free cash flows.
        """
        return self.ebit \
            - self.taxes \
            - self.depreciation_amortization \
            - self.impairment_charges \
            - self.capital_expenditures


class WACC:
    """ The Weighted Average Cost of Capital (WACC) is used to determine 
    the discounting rate in the discounted free cash flow model. 
    """
    def __init__(self,
                 country: str,
                 ref_risk_free_rate: float,
                 ref_long_term_growth: float,
                 country_risk_free_rate: float,
                 country_long_term_growth: float,
                 industry: str,
                 time_liquid_events: str,
                 company_size: float,
                 equity_weight: float,
                 cost_of_debt: float,
                 tax_rate: float
                 ):
        """ Instantiates the class

        Args:
            country (str): the country, should be part of the index of
                the country_risks dataframe
            ref_risk_free_rate (float): the reference risk free rate
            ref_long_term_growth (float): reference long-term growth
            country_risk_free_rate (float): country's risk free rate
            country_long_term_growth (float): country's long-term growth
            industry (str): the industry of the company, should be part
                of the index of the betas dataframe.
            time_liquid_events (str): time between liquid events. Should
                be part of the index of the illiquidity premiums dataset
            company_size (float): the size of the company expressed in 
                dollars.
            equity_weight (float): the equity weight ratio
            cost_of_debt (float): the total cost of debt
            tax_rate (float): the applicable tax rate
        """
        self.country = country
        self.ref_risk_free_rate = ref_risk_free_rate
        self.ref_long_term_growth = ref_long_term_growth
        self.country_risk_free_rate = country_risk_free_rate
        self.country_long_term_growth = country_long_term_growth
        self.industry = industry
        self.time_liquid_events = time_liquid_events
        self.company_size = company_size
        self.equity_weight = equity_weight
        self.cost_of_debt = cost_of_debt
        self.tax_rate = tax_rate
        # fixed illiquidity premium can change. Decision made here to
        # not define it as an input parameter but to hard-code it. This
        # can be changed without consequences.
        self.FIXED_ILLIQUIDITY_PREMIUM = 0.0082

    # Variables are often saved in properties because it is expected that the
    # way these values are calculated will change. This makes it easier to
    # implement these changes
    @property
    def risk_premium_country(self) -> float:
        """ A premium is used to account for risk differences between 
        countries. These risk premiums are based on the Damodaran risk
        tables: 
        https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html

        Returns:
            float: the risk premium of the country
        """
        return 0.1

    @property
    def equity_risk_premium(self) -> float:
        """ The equity risk premium combines the country premium with
        the risk free and long-term growth rates. This allows the 
        fund to correct for differences between the country of reference
        and the country of investment.

        Returns:
            float: the equity risk premium
        """
        return self.risk_premium_country \
            + self.ref_risk_free_rate \
            + self.country_long_term_growth \
            - self.ref_long_term_growth \
            - self.country_risk_free_rate

    @property
    def beta(self):  # WILL BE CHANGED
        return 0.8

    @property
    def illiquidity_premium(self):
        """ Since private equity invests in private (and therefore 
        illiquid) companies. An additional premium is calculated to 
        account for the illiquidity of the investment. This premium 
        is based on the time between liquid events (buy/sell) and 
        a fixed illiquidity premium.

        Returns:
            float: the illiquidity premium
        """
        return 0.15 + self.FIXED_ILLIQUIDITY_PREMIUM

    @property
    def size_premium(self):
        """ The size of the company in which the private equity fund
        invests, also determines how risky the investment is. Therefore,
        an additional size premium is taken into account.

        Returns:
            float: the size premium
        """
        # If the company is smaller than the minimum size specified
        # in the dataset, then we take the highest risk premium.
        if self.company_size < size_premiums['min'].min():
            return max(size_premiums.premium)/100
        # Else, we return the size premium based on the company size
        return 0.05

    @property
    def cost_of_equity(self):
        """ The cost of equity combines the information calculated above
        into a single metric that indicates the cost of equity.

        Returns:
            float: the cost of equity
        """
        return self.ref_risk_free_rate \
            + self.beta*self.equity_risk_premium \
            + self.illiquidity_premium \
            + self.size_premium

    @property
    def wacc(self):
        """ Combines the total cost of debt and the cost of equity in 
        a single metric which is then used in the discounted cash flow
        model to discount the free cash flows and the terminal value.

        Returns:
            float: the weighted average cost of capital
        """
        return self.cost_of_equity*self.equity_weight \
            + (1-self.tax_rate)*self.cost_of_debt*(1-self.equity_weight)


class DiscountedCashFlow:
    """
        The discounted free cash flow model calculates the enterprise 
        value based on the free cash flow, growth rate and, the weighted 
        average cost of capital.
    """
    def __init__(self,
                 free_cash_flow: np.ndarray,
                 wacc: float,
                 perpetual_growth_rate: float,
                 ) -> None:
        """ Instantiates the DiscountedCashFlow model

        Args:
            free_cash_flow (np.ndarray): the free cash flows expressed
                as a numpy array. Can most easily be obtained through
                the FreeCashFlow object with the property free_cash_flow
            wacc (float): the weighted average cost of capital. This
                variable can most easily be obtained through the wacc
                property of the WACC class. Note that the wacc should
                be strictly larger than the growth rate.
            perpetual_growth_rate (float): the growth rate for the 
                calculation of the terminal value. Note that the growth 
                rate must be smaller than the WACC.
        """
        assert wacc > perpetual_growth_rate, \
            f"Ensure that the wacc ({round(wacc, 4)}) is greater than the growth rate ({round(perpetual_growth_rate, 4)})." 
        self.free_cash_flow = free_cash_flow
        self.growth_rate = perpetual_growth_rate
        self.wacc = wacc
        # Variables for internal use
        self._time_horizon = len(free_cash_flow)

    @classmethod
    def from_objects(cls, 
                     free_cash_flow: FreeCashFlow, 
                     wacc: WACC, 
                     perpetual_growth_rate: float):
        """ Allows the creation of the DiscountedCashFlow model by
        directly providing the FreeCashFlow and WACC object.

        Args:
            free_cash_flow (FreeCashFlow): the FreeCashFlow object
            wacc (WACC): the WACC object
            perpetual_growth_rate (float): the perpetual growth rate
                used to calculate the terminal value.

        Returns:
            DiscountedCashFlow
        """
        free_cash_flow = free_cash_flow.free_cash_flow
        wacc = wacc.wacc
        return cls(free_cash_flow, wacc, perpetual_growth_rate)

    @property
    def terminal_value(self) -> float:
        """ Often, a firm explicitly forecasts the free cash flow over a 
        shorter horizon than the full horizon of the investment of the
        project. In this case, we estimate the remaining free cash flows 
        beyond the forecast horizon by including an additional, one-time 
        cash flow at the end of the forecast horizon. This additional 
        cash flow is called the terminal value (or continuation value).
        """
        # We determine the next free cash flow (at step time_horizon + 1)
        # to be equal to the last known free cash flow multiplied by
        # (1 + growth rate). The terminal value in this case is thus
        # represented by a one-time cash flow at time_horizon + 1.
        next_fcf: float = \
            self.free_cash_flow[-1] * (1 + self.growth_rate)
        # Once we have the next free cash flow, we transform this cash
        # flow into the terminal value by dividing it with the difference
        # between the wacc and the growth rate. This is based on the
        # assumption that the future cash flows will continue in 
        # perpetuity. 
        return next_fcf / (self.wacc - self.growth_rate)

    @property
    def discounted_free_cash_flows(self) -> float:
        """ The free cash flows need to be discounted to take into 
        account the time value of money. This function makes use of the 
        wacc to discount the free cash flows.

        Returns:
            np.ndarray: the free cash flows initially provided are now
                discounted to represent their present values.     
        """
        # The discount factors are based on the weighted average cost
        # of capital and the time until the cash flow takes place.
        discount_factors: np.ndarray = \
            (1 + self.wacc)**np.arange(self._time_horizon)
        return self.free_cash_flow / discount_factors

    @property
    def enterprise_value(self) -> float:
        """ The enterprise value is computed based on the discounted 
        free cash flows and the terminal value.
        """
        # The terminal value needs to be discounted. Since the terminal
        # value is basically a one-time cash flow at time_horizon + 1, 
        # we discount the value based on that number and the wacc.
        discounted_terminal_value: float = \
            self.terminal_value \
                / (1 + self.wacc)**(self._time_horizon + 1)
        # Now that everything has been discounted, the enterprise value
        # is simply the sum of all discounted free cash flows and the
        # discounted terminal value.
        return sum(self.discounted_free_cash_flows) \
            + discounted_terminal_value 

    ####
    # Sensitivity Analysis
    # --------------------
    # In the sensitivity analysis, we will explore the impact of 
    # chainging a parameter on the estimated enterprise/equity value.
    # The three parameters of interest are:
    #   (1) The wacc: the wacc determines the discount factors and 
    #          therefore strongly impacts the results
    #   (2) The perpetual growth rate: the perpetual growth rate, 
    #          together with the WACC determine the terminal value.
    #   (3) The EBIT margin: determines the free cash flows.

    def wacc_sensitivity(self,
                         lower_bound: float,
                         upper_bound: float,
                         correction=0) -> tuple:
        """ Implements the sensitivity analysis for the WACC.

        Args:
            lower_bound (float): the minimum wacc that is considered. 
                Should be greater than 0 
                Should always be greater than the growth rate
            upper_bound (float): the maximum wacc that is considered
            correction (float, optional): The EV-EQ Bridge. Defaults to 0.
        
        Returns:
            tuple: a tuple containing the equity values and the 
                associated waccs.
        """
        assert 0 < lower_bound < upper_bound, \
            f"Ensure that the lower bound ({lower_bound}) is stricly smaller than the upper bound ({upper_bound}) and greater than 0"
        assert lower_bound > self.growth_rate, \
            f"Ensure that the lower bound is greater than the growth rate ({self.growth_rate})."
        options = np.arange(start=lower_bound, stop=upper_bound,
                            step=(upper_bound-lower_bound)/100)
        enterprise_values = np.array([
            DiscountedCashFlow(self.free_cash_flow,
                               option,
                               self.growth_rate).enterprise_value
            for option in options])
        equity_values = enterprise_values - correction
        return equity_values, options

    def growth_sensitivity(self,
                           lower_bound: float,
                           upper_bound: float,
                           correction=0) -> tuple:
        """ Implements the sensitivity analysis for the perpetual growth
        rate.

        Args:
            lower_bound (float): the minimum growth that is considered. 
            upper_bound (float): the maximum growth that is considered
            correction (float, optional): THE EV-EQ Bridge. Defaults to 0.
        
        Returns:
            tuple: a tuple containing the equity values and the 
                associated growth rates.
        """
        assert lower_bound < upper_bound, \
            "Ensure that the lower bound is stricly smaller than the upper bound"
        assert upper_bound < self.wacc, \
            f"Ensure that the upper bound is smaller than the wacc ({self.wacc})."
        options = np.arange(start=lower_bound, stop=upper_bound,
                            step=(upper_bound-lower_bound)/100)
        enterprise_values = np.array([
            DiscountedCashFlow(self.free_cash_flow,
                               self.wacc,
                               option).enterprise_value
            for option in options])
        equity_values = enterprise_values - correction
        return equity_values, options

    def ebit_sensitivity(self,
                         lower_bound: float,
                         upper_bound: float,
                         tax_rate: float,
                         depreciation_amortization: float,
                         impairment_charges: float,
                         capital_expenditures: float,
                         company_revenue: np.ndarray,
                         correction: float=0) -> tuple:
        """ Implementes the sensitivity analysis for the ebit margin.

        Args:
            lower_bound (float): the minimum growth that is considered. 
            upper_bound (float): the maximum growth that is considered
            tax_rate (float): the applicable tax rate for the company. 
                Note: assumed to be a constant i.e. no progressive taxes
            depreciation_amortization (np.ndarray): the depreciation and 
                amortization correction applied to calculate the free
                cash flows.
            impairment_charges (np.ndarray): the impairment charges
                correction applied to calculate the free cash flows.
            capital_expenditures (np.ndarray): the capital expenditures 
                correction applied to calculate the free cash flows.
            company_revenue (np.ndarray): the company revenues used
                as the starting point of the calculations.
            correction (float, optional): THE EV-EQ Bridge. Defaults to 0.

        Returns:
            tuple: a tuple containing the equity values and the 
                associated ebit margins.
        """
        assert lower_bound < upper_bound, \
            "Ensure that the lower bound is stricly smaller than the upper bound"
        options = np.arange(start=lower_bound, stop=upper_bound,
                            step=(upper_bound-lower_bound)/100)
        enterprise_values = np.array([
            DiscountedCashFlow(
                FreeCashFlow.from_ebit_margin(
                    tax_rate=tax_rate,
                    depreciation_amortization=depreciation_amortization,
                    impairment_charges=impairment_charges,
                    capital_expenditures=capital_expenditures,
                    ebit_margin=option,
                    company_revenue=company_revenue
                ).free_cash_flow,
                self.wacc,
                self.growth_rate).enterprise_value
            for option in options])
        equity_values = enterprise_values - correction
        return equity_values, options


class DiscountedCashFlowReversed:
    """ The DiscountedCashFlowReversed class implemented the reverse
    calculations of the DiscountedCashFlow model. In the reversed 
    calculations, the starting point is the enterprise value and the 
    goal is to calculate values for the main parameters that result
    in the specified enterprise value.

    All of these reverse calculations make use of a very simple 
    optimization algorithm that is sufficiently fast. The main idea is
    that we always take a small step in the direction of the correct
    solution until we are in the correct solution. The direction
    of the step is based on the sign of the derivative (which does not 
    need to be calculated because we know this sign), the size of the
    step is predefined but decreases when we are moving around the
    optimal solution. It's very simple, but it works fast.
    """
    def __init__(self,
                 target_value: float,
                 epsilon: float = 1e-3,
                 correction: float = 0.0) -> None:
        """ Instantiates the class

        Args:
            target_value (float): the target enterprise value
            epsilon (float, optional): the mistake that is allowed. 
                Defaults to 1e-3.
            correction (float, optional): THE EV-EQ Bridge. Defaults to 0.
        """
        self.target_value = target_value
        self.epsilon = epsilon
        self.correction = correction
        # For internal use
        # These initial guesses can have a large influence on the speed
        # of conversion. Therefore, these have been hard coded here and
        # are not passed as parameters. 
        self._initial_guesses = {
            "growth_rate": 0.02, "wacc": 0.10, "ebit_margin": 0.10
        }
        # The step size determines how quicly we converge to the solution
        # even though it is specified as a constant here, the step size
        # is actually dynamic because it decreases when we are moving
        # closer to the solution.
        self._step_size = 0.01

    ####
    # Private methods
    ####
    def __set_starting_values(self, 
                              to_be_estimated: str, 
                              parameters: dict) -> dict:
        """ Gets the starting estimates for the optimization algorithm

        Args:
            to_be_estimated (str): the variable that will be estimated
                by the optimization algorithm
            parameters (dict): the known parameters that are provided
                as inputs to the optimization algorithm

        Returns:
            dict: a dictionary in which the keys correspond to the 
                parameters (including the parameter that needs to be
                estimated) and the starting values as the values.
        """
        starting_values: dict = {}
        # For the parameter that needs to be estimated, we define the
        # starting value as the initial guess
        starting_values[to_be_estimated] = \
            self._initial_guesses[to_be_estimated]
        # For the other parameters, the starting values are simply the 
        # ones provided by the user.
        for parameter, value in parameters.items():
            starting_values[parameter] = value
        return starting_values

    def __estimate_enterprise_value(self, 
                                    to_be_estimated: str,
                                    values: dict) -> float:
        """ Estimates the enterprise value based on the provided values.

        Args:
            to_be_estimated (str): the parameter that is estimated by
                the model
            values (dict): the values of the input parameters

        Returns:
            float: the enterprise value
        """
        # In case the ebit margin is estimated, we first need to 
        # calculate the free cash flows
        if to_be_estimated == 'ebit_margin':
            free_cash_flow = FreeCashFlow.from_ebit_margin(
                tax_rate=values['tax_rate'],
                depreciation_amortization=values['depreciation_amortization'],
                impairment_charges=values['impairment_charges'],
                capital_expenditures=values['capital_expenditures'],
                ebit_margin=values['ebit_margin'],
                company_revenue=values['company_revenue']
            ).free_cash_flow
        else:
            free_cash_flow = values["free_cash_flow"]
        return DiscountedCashFlow(
            free_cash_flow=free_cash_flow,
            wacc=values['wacc'],
            perpetual_growth_rate=values["growth_rate"]
        ).enterprise_value

    def __optimize(self, 
                   to_be_estimated: str, 
                   parameters: dict,
                   derivative_positive=True) -> tuple:
        """ Performs the optimization to find the parameter values.

        Args:
            to_be_estimated (str): the parameter that needs to be 
                estimated
            parameters (dict): additional parameters that are required
                to estimate the parameter of interest
            derivative_positive (bool, optional): indicates whether
                the derivative is positive. This is used to determine
                the direction of the steps taken towards the solution.
                Defaults to True.

        Returns:
            tuple: a tuple containing the parameter of interest's value
                and the associated enterprise value
        """
        assert to_be_estimated in self._initial_guesses, \
            "Make sure that the to_be_estimated parameter is valid."
        # Setting some useful variables
        _original_step_size, estimated_EV = self._step_size, np.Inf
        _previous_mistake_positive = True
        values = self.__set_starting_values(
            to_be_estimated=to_be_estimated, parameters=parameters)
        # As long as the mistake is not within our margin of error, we
        # continue
        while np.abs(estimated_EV - self.target_value) >= self.epsilon:
            mistake = estimated_EV - self.correction - self.target_value
            if (_previous_mistake_positive and mistake < 0) \
                    or (not _previous_mistake_positive and mistake > 0):
                # Half the step size if the solution is in between
                # the current and previous estimate
                self._step_size = self._step_size / 2
            # Determine the direction of the step
            if (mistake > 0) == derivative_positive:
                values[to_be_estimated] -= self._step_size
            else:
                values[to_be_estimated] += self._step_size
            # Re-estimate the enterprise value and save whether the 
            # current mistake is positive or negative.
            estimated_EV = self.__estimate_enterprise_value(
                to_be_estimated=to_be_estimated, values=values
            )
            _previous_mistake_positive = mistake > 0
        self._step_size = _original_step_size
        return values[to_be_estimated], estimated_EV


    ####
    # Public methods
    ####
    def get_growth_estimate(self, 
                            free_cash_flow: np.ndarray, 
                            wacc: float) -> tuple:
        """ Estimates the growth rate based on the target enterprise
        value, the free cash flows and the wacc.

        Args:
            free_cash_flow (np.ndarray): the free cash flow. Most easily
                obtained via the FreeCashFlow object.
            wacc (float): the weighted average cost of capital. Most
                easily obtained via the WACC object.

        Returns:
            tuple: the estimated growth rate and the associated 
                enterprise value
        """
        growth, estimated_EV = self.__optimize(
            to_be_estimated="growth_rate",
            parameters={"free_cash_flow": free_cash_flow,
                        "wacc": wacc},
            derivative_positive=True
        )
        return growth, estimated_EV

    def get_wacc_estimate(self, 
                          free_cash_flow: np.ndarray, 
                          growth_rate: float) -> tuple:
        """ Estimates the wacc based on the target enterprise value, 
        the free cash flow and the growth rate.

        Args:
            free_cash_flow (np.ndarray): the free cash flow. Most easily
                obtained via the FreeCashFlow object.
            growth (float): the perpetual growth rate used for the 
                calculation of the terminal value.

        Returns:
            tuple: the estimated wacc and the associated enterprise
                value
        """
        wacc, estimated_EV = self.__optimize(
            to_be_estimated="wacc",
            parameters={"free_cash_flow": free_cash_flow,
                        "growth_rate": growth_rate},
            derivative_positive=False
        )
        return wacc, estimated_EV

    def get_ebit_estimate(self, 
                          wacc: float, 
                          growth_rate: float, 
                          tax_rate: float,
                          depreciation_amortization: np.ndarray,
                          impairment_charges: np.ndarray,
                          capital_expenditures: np.ndarray,
                          company_revenue: np.ndarray) -> tuple:
        """ Estimates the ebit margin based on the target enterprise 
        value, the wacc, the growth rate, and the other parameters
        required for the calculation of the free cash flows.

        Args:
            wacc (float): the weighted average cost of capital. Most
                easily obtained via the WACC object.
            growth (float): the perpetual growth rate used for the 
                calculation of the terminal value.
            tax_rate (float): the applicable tax rate for the company. 
                Note: assumed to be a constant i.e. no progressive taxes
            depreciation_amortization (np.ndarray): the depreciation and 
                amortization correction applied to calculate the free
                cash flows.
            impairment_charges (np.ndarray): the impairment charges
                correction applied to calculate the free cash flows.
            capital_expenditures (np.ndarray): the capital expenditures 
                correction applied to calculate the free cash flows.
            company_revenue (np.ndarray): the company revenues used
                as the starting point of the calculations.

        Returns:
            tuple: the estimated ebit margin and the associated 
                enterprise value
        """
        ebit, estimated_EV = self.__optimize(
            to_be_estimated="ebit_margin",
            parameters={"tax_rate": tax_rate,
                        "depreciation_amortization": depreciation_amortization,
                        "impairment_charges": impairment_charges,
                        "capital_expenditures": capital_expenditures,
                        "company_revenue": company_revenue,
                        "wacc": wacc,
                        "growth_rate": growth_rate},
            derivative_positive=True
        )
        return ebit, estimated_EV

    def get_wacc_growth_estimates(self, 
                                 free_cash_flow: np.ndarray,
                                 min_growth: float=-0.1,
                                 max_growth: float=0.1,
                                 return_ev_matrix: bool=False) -> tuple:
        """ Estimates the possible solutions for the wacc and the 
        growth rate based on the target enterprise value and the 
        free cash flows. Note that there exist an infinite number of 
        solutions. Therefore, the domain is bounded.

        Args:
            free_cash_flow (np.ndarray): the free cash flow. Most easily
                obtained via the FreeCashFlow object.
            min_growth (float, optional): the lower bound for the growth
                rate. Defaults to -0.1.
            max_growth (float, optional): the upper bound for the growth
                rate. Defaults to 0.1.
            return_ev_matrix (bool, optional): determines whether the 
                enterprise value matrix is returned. This enterprise
                value matrix can be used to construct contour plots.
                Defaults to False.

        Returns:
            tuple: the solution for the growth_rates and waccs. If
            return_ev_matrix is True, then the enterprise value matrix
            is also returned.
        """
        # Realize that for each valid value of the growth rate, there is
        # a wacc that returns the target enterprise value. Therefore,
        # we simply need to loop over the possible values for growth 
        # rates and calculate the associated wacc using the method
        # defined above.
        growth_rates = list(np.arange(min_growth, max_growth, step=0.001))
        waccs = [self.get_wacc_estimate(free_cash_flow, growth)[0]
                    for growth in growth_rates]
        if return_ev_matrix:
        # To construct the contours, we also perform a calculation of 
        # the enterprise value for each of the combinations of the growth
        # and the wacc
            ev_matrix = [
                [abs(DiscountedCashFlow(free_cash_flow, wacc, growth).enterprise_value - self.target_value)
                        if growth < wacc - 0.01 else 0
                    for growth in growth_rates]
                    for wacc in waccs
            ]
            return growth_rates, waccs, ev_matrix
        else:
            return growth_rates, waccs