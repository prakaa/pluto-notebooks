### A Pluto.jl notebook ###
# v0.19.17

using Markdown
using InteractiveUtils

# ╔═╡ bbe08740-1e29-4f63-a755-0bf08d3df898
begin
	using PlutoUI
	TableOfContents(indent=true, depth=3)
end

# ╔═╡ 05e4f039-9a2a-4197-b25f-0210ce0baa6a
begin
	# for handling model output data
	using DataFrames
	# our linear solver whcih also returns duals (shadow prices)
	using GLPK
	# mathematical programming toolkit
	using JuMP
	# efficient small data structures
	using StaticArrays
	# plotting packages
	using Plots
	using VegaLite
end

# ╔═╡ c2e7dc36-b8fc-471c-a6f3-f80aef467eb0
md"# Multi vs. Single Period Dispatch

In this [Pluto.jl notebook](https://www.youtube.com/watch?v=IAF8DjrQSSk), I dig into the dispatch and pricing implications of multi-period economic dispatch. In the Australian National Electricity Market, economic dispatch is single period - that is, the optimisation problem (which we can simplify to *meeting demand at lowest generation cost*) is solved for a single interval. In multi-period dispatch, the optimiser finds a solution such that demand over several intervals is met at lowest cost.

A couple of the Independent System Operators in the U.S. (e.g. CAISO and MISO) have implemented some form of multi-period dispatch ([see this excellent overview by Dane Schiro from ISO-NE](https://www.ferc.gov/sites/default/files/2020-08/M1-3_Schiro.pdf)).

This notebook was inspired by the papers below:
- E. Ela and M. O'Malley, [Scheduling and Pricing for Expected Ramp Capability in Real-Time Power Markets](https://ieeexplore.ieee.org/document/7192736), IEEE Transactions on Power Systems
- Jacob Mays, [Missing incentives for flexibility in wholesale electricity markets](https://doi.org/10.1016/j.enpol.2020.112010), Energy Policy

## A little intro to multi-period dispatch
Why are we interested in multi-period dispatch? It comes down to how it can address dispatch variability (expected changes in supply/demand). Multi-period dispatch can ramp resources in response to demand changes. 

### Single vs multi-period ramping constraints
- Single period dispatch has what we'll call *backwards-looking* ramping constraints: *is the ramp from the operating point at the end of the previous interval to the target at the end of this interval feasible for that resource?* 
- Multi-period dispatch has backwards-looking ramping constraints, but also incorporates *forward-looking* ramping constraints: *is the ramp from the target at the end of this interval to the target at the end of the following interval feasible?*

The **incorporation of forward-looking ramping constraints has interesting pricing implications we'll look at in this notebook**.

## Some things to take note of

1. Our example **assumes perfect foresight**. That is, multi-period dispatch can be used to help manage variability but it **may not ensure a feasible dispatch outcome if there is uncertainty (e.g. renewable energy or demand forecast errors)**.
    - This becomes more relevant the greater the number of periods that the multi-period dispatch solves for.
2. We solve each *horizon* (the number of periods solved simultaneously in a multi-dispatch problem) sequentially. For example, for a 2-period dispatch, we solve periods 1 & 2 simultaneously, then periods 3 & 4 simultaneously, etc.
    - However, rather than solving problems sequentially, ISO markets typically take a *rolling-horizon* approach. 
      - For example, for a 3-period dispatch problem, periods 1, 2 & 3 are solved simultaneously. Then as we near the start of dispatch interval 2, periods 2, 3 & 4 are solved simultaneously, etc. A rolling horizon approach enables the latest information to be used in dispatch.
3. [As discussed by Schiro](https://www.ferc.gov/sites/default/files/2020-08/M1-3_Schiro.pdf), where multi-period dispatch has been implemented, only the price of the first interval is typically considered *binding* (i.e. relevant for market settlement); the prices of the following intervals are considered to be *advisory*. This is a practical but incomplete implementation of multi-period dispatch as the resulting series of prices, not a single price, ensure there is sufficient incentive for participants to meet system needs and comply with dispatch instructions.
"

# ╔═╡ 249be87e-08a9-49c1-98e8-81a421f1095e
md"## Model Code

I've written the code for the single and multi-period dispatch models below. If it's your jam, you can have a look at how I've set the model up. Even if you're not super familiar with Julia, models written in its mathematical programming package ([JuMP](https://jump.dev/JuMP.jl/stable/)) somewhat resemble how you'd write out an optimisation problem using mathematical notation - a big plus for readability!

If you're not really interested in this, head down to the the **Input Data** section.
"


# ╔═╡ 85d9814c-8523-4be3-b279-918eef5a55bb
md"### Packages

Use packages for handling data, writing our optimisation model, solving the optimisation model and plotting the results
"

# ╔═╡ 0d561b8f-f79e-4587-a602-21c5f6101a8a
md"### Functions and Structs for the Simulations"

# ╔═╡ b4260f67-110c-4d90-92aa-43756fd32ca6
md"#### Defining a Generator Data Structure"

# ╔═╡ 4c33ffd2-bda9-4ea9-961b-c9ce9f0821d6
begin
	struct Generator{T <: Float64}
	    name::String
	    min_gen::T
	    max_gen::T
	    ramp_up::T
	    ramp_down::T
	    offer::T
	end
	
	"""
	    Generator(name::String; min_gen::Float64=0.0, max_gen::Float64=0.0,
	              ramp_up::Float64=9999.0, ramp_down::Float64=9999.0, offer::Float64=0.0)
	
	Creates a Generator
	
	##### Arguments
	- `name::String`: Name of the generator. Must be unique.
	- `min_gen::Float64`: Minimum generation in MW.
	- `max_gen::Float64`: Maximum generation in MW.
	- `ramp_up::Float64`: Ramp up rate in MW/min.
	- `ramp_down::Float64`: Ramp down rate in MW/min.
	- `offer::Float64`: Offer for energy in \$\$/MW/hr.
	
	"""
	function Generator(name::String; min_gen::Float64=0.0, max_gen::Float64=0.0,
	                   ramp_up::Float64=9999.0, ramp_down::Float64=9999.0,
	                   offer::Float64=0.0)
	    @assert(max_gen ≥ min_gen, "Maximum generation should be greater than 					minimum")
	    @assert(min_gen ≥ 0.0, "Minimum generation should be greater than 0")
	    @assert(ramp_up ≥ 0.0, "Ramp up should be greater than 0")
	    @assert(ramp_down ≥ 0.0, "Ramp down should be greater than 0")
	    @assert(offer ≥ 0.0, "Offer price should be greater than 0")
	    return Generator(name, min_gen, max_gen, ramp_up, ramp_down, offer)
	end
end

# ╔═╡ 99681fe9-8f7e-4681-976b-e44ebb9bcc5b
md"#### Functions for simulations"

# ╔═╡ 689eae05-e366-4595-8fcd-923e6d95dc16
"""
    initialise_singleperiod(model::Model, t::Int64, demand,
                            generator_ids::Dict{Symbol, Generator{Float64}};
                            generator_ini_conds::Dict{Symbol, Float64}=Dict{Symbol,Float64}())
For a given (5-minute) interval `t` (i.e. the interval *number* is `t`), create a single-period dispatch model.
If `t > 1`, the function expects initial conditions (i.e. MW generation) provided in
`generator_ini_conds` so that a backward-looking ramp constraint between `t-1` and `t` can be added to the model.
"""
function initialise_singleperiod(model::Model, t::Int64, demand::SVector,
                                 generator_ids::Dict{Symbol, Generator{Float64}};
                                 generator_ini_conds::Dict{Symbol, Float64}=Dict{Symbol, Float64}())
    @variable(model,
              generator_ids[i].min_gen
              <= GENERATION[i=keys(generator_ids)]
              <= generator_ids[i].max_gen)
    @variable(model, UNSERVED ≥ 0.0)
	# Ramp constraints
    ## convert MW/min to MW by multiplying by 5 (minutes)
    if t > 1
		@constraint(model, RampUp[i=keys(generator_ids)],
					GENERATION[i] - generator_ini_conds[i]
					≤ generator_ids[i].ramp_up * 5.0)
        @constraint(model, RampDown[i=keys(generator_ids)],
                    GENERATION[i] - generator_ini_conds[i]
                    ≥ -generator_ids[i].ramp_down * 5.0)
    end
	# Balance constraint
    @constraint(model, Balance,
                sum(GENERATION[i] for i in keys(generator_ids))
                + UNSERVED  == demand[t])
	# Objective function with unserved
    @objective(model, Min, sum(generator_ids[i].offer * GENERATION[i]
                               for i=keys(generator_ids)) + sum(15000.0 * UNSERVED))
    return model
end

# ╔═╡ f1591a89-37c6-4f88-99e3-64d8cb415f72
"""
    initialise_multiperiod(model::Model, t_init::Int64, t_end::Int64, demand::SVector,
                           generator_ids::Dict{Symbol, Generator{Float64}};
						   generator_ini_conds::Dict{Symbol, Float64}=Dict{Symbol, Float64}())

Creates a multi-period dispatch model for all (5-minute) dispatch intervals from `t_init` to `t_end`. Forward-looking ramping constraints apply from `t` to `t_end`.
If `t_init > 1` (i.e. `t_init` is not the first interval), the function expects initial conditions (i.e. MW generation) provided in
`generator_ini_conds` so that a ramp constraint between `t_init-1` and `t_init` can be added to the model (i.e. backward-looking ramp constraint).
"""
function initialise_multiperiod(model::Model, t_init::Int64, t_end::Int64, 			
								demand::SVector,
                                generator_ids::Dict{Symbol, Generator{Float64}};
								generator_ini_conds::Dict{Symbol, Float64}=Dict{Symbol, Float64}())
    @assert(t_init ≤ t_end, "End time must be ≥ start time")
    @variable(model,
              generator_ids[i].min_gen
              <= GENERATION[i=keys(generator_ids), t=t_init:t_end]
              <= generator_ids[i].max_gen)
    @variable(model, UNSERVED[t=t_init:t_end] ≥ 0.0)
	# Ramp constraints (backwards and forward looking)
    ## convert MW/min to MW by multiplying by 5 (minutes)
	if t_init > 1
		@constraint(model, BwdRampUp[i=keys(generator_ids), t=t_init],
		                             GENERATION[i, t] - generator_ini_conds[i]
		                             ≤ generator_ids[i].ramp_up * 5.0)
	    @constraint(model, BwdRampDown[i=keys(generator_ids), t=t_init],
	                                   GENERATION[i, t] - generator_ini_conds[i]
	                                   ≥ -generator_ids[i].ramp_down * 5.0)
	end
    @constraint(model, FwdRampUp[i=keys(generator_ids), t=t_init:t_end-1],
                                 GENERATION[i, t+1] - GENERATION[i, t]
                                 ≤ generator_ids[i].ramp_up * 5.0)
    @constraint(model, FwdRampDown[i=keys(generator_ids), t=t_init:t_end-1],
                                   GENERATION[i, t+1] - GENERATION[i, t]
                                   ≥ -generator_ids[i].ramp_down * 5.0)
	# Balance constraint
    @constraint(model, Balance[t=t_init:t_end],
                               sum(GENERATION[i, t] for i in keys(generator_ids))
                               + UNSERVED[t]  == demand[t])
	# Objective function with unserved
    @objective(model, Min, sum(generator_ids[i].offer * GENERATION[i, t]
                               for i=keys(generator_ids), t=t_init:t_end)
                           + sum(15000.0 * UNSERVED[t] for t=t_init:t_end))
    return model
end

# ╔═╡ 9e76c37f-c4a4-4b12-8f4a-7c12133f19ac
begin
	html"""<a class="anchor" id="input"></a>"""
	md"## Input Data"
end

# ╔═╡ 3a298fac-8503-46e8-8747-56b93b8564b5
md"#### Generators

Here we create a few generators. The operational envelope of each generator (i.e. the feasible operational space defined by generation constraints) does not necessarily correspond to typical values for a generator of that technology type. Rather, we mimic the *relative* operational envelopes of generation technologies. 

We have:
1. A less flexible coal plant with a higher minimum stable level. This plant represents the bulk of the system's generation capacity. The coal plant offers energy into the market at its fuel price. 
2. A more flexible peaker plant with a lower minimum stable level. This peaker plant offers energy to the market at a premium of \$1000/MW/hr.
3. A very flexible wind farm that offers energy into the market at no cost. Note that we assume that the wind farm is able to generate freely in its capacity range (i.e. wind generation is not constrained by wind availability). As a result, the wind farm can ramp almost instantaneously within its generation limits.
"

# ╔═╡ 9f8913c5-6aa4-4137-9ab7-5f4313371b7d
"""
    create_generators()

Creates a coal, wind and (gas) peaker generator, with the following properties:

| Generator | Generation Range (MW) | Ramp Up/Down (MW/min) | Offer (\$\$/MW/hr)|
|-----------|-----------------------|-----------------------|-------------------|
| Coal      |      100-2,000   	   	|      30/100 			|    40      		|
| Wind      |      0-300         	|      9999/9999		|    0       		|
| Peaker    |      10-300           |      60/100           |    1000    		|

"""
function create_generators()
    coal = Generator("Coal", min_gen=100.0, max_gen=2000.0, ramp_up=30.0, 								 ramp_down=100.0, offer=40.0)
    wind = Generator("Wind", min_gen=0.0, max_gen=300.0, ramp_up=9999.0,
                     ramp_down=9999.0, offer=0.0)
    peaker = Generator("Peaker", min_gen=10.0, max_gen=300.0, ramp_up=60.0, 							    ramp_down=100.0, offer=1000.0)
    generators_ids = Dict(Symbol(gen.name) => gen for (i, gen)
                          in enumerate((coal, wind, peaker)))
    return generators_ids
end

# ╔═╡ 537cb82f-825d-46ae-a69a-d8a20493a689
"""
	run_singleperiod(demand::SVector)

Initialises and solves a series of single-period dispatch problems, given `demand` (electricity demand every 5 minuts).

Returns DataFrames for market price, generation and unserved energy.
"""
function run_singleperiod(demand::SVector)
	"""
	Extracts price, generation and USE data from model and adds to preovided DataFrames.
	"""
	function extract_data!(prices::DataFrame, generation::DataFrame,
						   use::DataFrame, t_init::Int64; model::Model)
		dualval = dual.(model[:Balance])
		gen_data = value.(model[:GENERATION])
		unserved = value.(model[:UNSERVED])

		prices = vcat(prices, DataFrame(:interval=>t_init,
										:prices=>dualval))
		generation = vcat(generation, DataFrame(
			:generator=>gen_data.axes[1],
			:interval=>repeat([t_init], length(generator_ids)),
			:generation=>gen_data.data
			))
		use = vcat(use, DataFrame(:interval=>t_init, :unserved=>unserved))
		return prices, generation, gen_data, use
	end
	
    generator_ids = create_generators()
    (prices, generation, use) = (DataFrame(), DataFrame(), DataFrame())
	ini_conds = Dict{Symbol, Float64}()
    for t_init in 1:length(demand)
        model = Model(GLPK.Optimizer)
        if t_init == 1
            model = initialise_singleperiod(model, t_init, demand, generator_ids)
        else
            model = initialise_singleperiod(model, t_init, demand, generator_ids,
                                            generator_ini_conds=ini_conds)
        end
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL && has_duals(model)
			
			(prices, generation, gen_data, use) = extract_data!(prices, generation,
																				use, t_init, 						model=model)
			for (sym, gen) in zip(gen_data.axes[1], gen_data.data)
				ini_conds[sym] = gen
			end
		elseif !has_duals(model)
            println("Change solver to one that calculates duals")
        else
            println("Error in optimising model")
        end
    end
    return prices, generation, use
end

# ╔═╡ 6b937c1a-f6e8-4a57-99ab-cd4391f24fef
"""
    run_multiperiod(demand::SVector, t_horizon::Int64)

Initialises and solves a series of multi-period dispatch models/simulations given `demand`, which contains electricity demand every 5 minutes. The **horizon** (the number of 5-minute intervals solved simultaenously in a multi-period dispatch problem) of each model/simulation is defined by `t_horizon`, which must be a factor of
`length(demand)`.

Returns DataFrames for market price, generation and unserved energy.
"""
function run_multiperiod(demand::SVector, t_horizon::Int64)
	"""
	Extracts price, generation and USE data from model and adds to preovided DataFrames.
	"""
	function extract_data!(prices::DataFrame, generation::DataFrame, use::DataFrame,
						   ini_conds::Dict{Symbol, Float64},
						   t_init::Int64, t_end::Int64; model::Model)
		dual_array = Array(dual.(model[:Balance]))
		gen_array = value.(model[:GENERATION].data)
		use_array = Array(value.(model[:UNSERVED]))
		prices = vcat(prices, DataFrame(:interval=>t_init:t_end,
										:prices=>dual_array))
		generation = vcat(generation, DataFrame(
			:generator=>repeat(model[:GENERATION].axes[1], t_end - t_init + 1),
			:interval=>repeat(t_init:t_end, inner=size(gen_array, 1)),
			:generation=>reshape(gen_array, length(gen_array))
			))
		use = vcat(use, DataFrame(:interval=>t_init:t_end, :unserved=>use_array))
		last_gen = value.(model[:GENERATION])[:, end]
		for (gen, val) in zip(last_gen.axes[1], last_gen.data)
			ini_conds[gen] = val
		end
		return prices, generation, ini_conds, use
	end
	
    @assert(length(demand) % t_horizon == 0, "t_horizon must be a factor of demand")
    generator_ids = create_generators()
    (prices, generation, use) = (DataFrame(), DataFrame(), DataFrame())
	t_inits = collect(1:t_horizon:length(demand))
	ini_conds = Dict{Symbol, Float64}()
    for t_init in t_inits
        t_end = t_init + t_horizon - 1
        model = Model(GLPK.Optimizer)
        model = initialise_multiperiod(model, t_init, t_end, demand, generator_ids;
									   generator_ini_conds=ini_conds)
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL && has_duals(model)
            (prices, generation, ini_conds, use) = extract_data!(prices, generation, 																 use, ini_conds,
																				 t_init, t_end,
																				 model=model)
        elseif !has_duals(model)
            println("Change solver to one that calculates duals")
        else
            println("Error in optimising model")
        end
    end
    return prices, generation, use
end

# ╔═╡ d861def0-9513-4da1-8ee8-a266868c4447
begin
	gens = create_generators()
	gendata = DataFrame([:fueltype=>[gens[key].name for key in keys(gens)],
						 :capacity=>[gens[key].max_gen for key in keys(gens)]])
	gendata |> @vlplot(
		:bar, x={:capacity, axis={title="Capacity (MW)"}}, y="sum()", 
		color={
			:fueltype, 
			scale={
				domain=["Coal", "Peaker", "Wind"], 
				range=["#808080","#deae93","#69b008"],
				},
			legend={title="Fuel Type"}
			},
		width=500, height=200
		)
end

# ╔═╡ ea7ce8b8-f2a6-4fd3-b5aa-71588293b271
md"#### Demand
Here we initialise a demand time series across intervals 1 to 10. To explore dispatch and pricing outcomes of ramp events in single and multi-period processes, this demand time series contains significant upward and downward ramps  (i.e. in the order of system ramping capabilities).
"

# ╔═╡ 874cd3d3-2648-4b9f-82cc-c34485b6848f
begin
	const demand = @SVector[880.0, 1200.0, 1000.0, 1300.0, 1500.0,
	                        2000.0, 1000.0, 800.0, 900.0, 1000.0]
	plot(1:10, demand, ylabel="MW", xlabel="Interval", title="System Demand", 			 legend=false)
end

# ╔═╡ 3fca603b-21e6-4aaf-83b6-17b60d8e9218
md"The chart below shows demand ramping against thermal and total generation ramping capabilities. It's worth highlighting that the demand ramp exceeds thermal generation ramping capabilities between the end of interval 5 and the end of interval 6. 

To meet this ramp and avoid unserved energy (set at \$15,000/MW/hr), the system will need to appropriately 'position' thermal generation and the wind farm in preceding intervals. Specifically, if the system maximises the use of cheaper generation (wind) in intervals prior to interval 5, the system may be ill-positioned to respond to the high demand ramp."

# ╔═╡ 9c29a181-9a29-4fec-85f0-ad87badf0445
begin
	r = plot(["$i to $(i+1)" for i in range(1, 9, step=1)],
			 diff(demand) ./ 5.0, label="Demand Ramp", ylabel="Ramp (MW/min)",
			 legend=:left)
	plot!(r, 1:9, repeat([90.0], 9), label="Thermal Total Ramp Up", ls=:solid, lw=2,
		  color=:red)
	plot!(r, 1:9, repeat([390.0], 9), label="Thermal + Wind Total Ramp Up", ls=:dash, 	  lw=2, color=:red)
	plot!(r, 1:9, repeat([-200.0], 9), label="Thermal Total Ramp Down", ls=:solid, 		  lw=2, color=:purple)
	plot!(r, 1:9, repeat([-500.0], 9), label="Thermal + Wind Total Ramp Down", 			  ls=:dash, lw=2, legend=:outerbottom, color=:purple)
	title!("Demand Ramping vs. Generator Ramp Rates")
	annotate!([(4.75, -350, "Wind is assumed to be capable of very\nhigh ramp rates within its capacity limits", 8)])
end

# ╔═╡ f00e8a37-a2ba-4744-8abc-675b51f774d4
md"### Run different dispatch processes

Below, we run the following:

1. Single period (each 5-minute interval solved separately)
2. 2-period dispatch (10 minutes solved simultaneously)
3. 5-period dispatch (25 minutes solved simultaneously)
4. 10-period dispatch (50 minutes solved simultaneously)

Dispatch process 1 (single-period) involves backwards-looking ramping constraints (i.e. is the ramp feasible from `t-1` to `t`?), whereas dispatch processes 2-4 involve both backwards and forward-looking ramping constraints (i.e. are the ramps from `t-1` to `t`, `t` to `t+1`, `t+1` to `t+2`, etc. feasible?).
"

# ╔═╡ 5bd442ad-c171-40f1-ae98-6548667a8b1f
begin
	(prices_1, gen_1, usmw_1) = run_singleperiod(demand)
	(prices_2, gen_2, usmw_2) = run_multiperiod(demand, 2)
	(prices_5, gen_5, usmw_5) = run_multiperiod(demand, 5)
	(prices_10, gen_10, usmw_10) = run_multiperiod(demand, 10)
end;

# ╔═╡ 56cb2867-4c2c-4f30-b8af-29fae8825f13
md"## Results"

# ╔═╡ 27ae61bb-5333-4894-9a10-c34ba46ae2f4
md"##### Dispatch"

# ╔═╡ 27c881e5-ad66-42e5-88eb-3b4e28597593
begin
	demandata = DataFrame(:interval=>1:10, :demand=>demand)
	gen_1 |>
	@vlplot(x="interval:n", width=600, height=300) + 
	@vlplot(
		mark={:area},
		y={:generation, title="Generation (MW)"},
		color={:generator, legend={title="Generator Type"}},
		title="1-period dispatch"
	) +
	(
		demandata |>
		@vlplot(
			mark={:line, color=:black,strokeDash=2}, 
			x="interval:n", y=:demand
		)
	)
end

# ╔═╡ dfc1f6f9-b436-4a5c-914b-790dc3e8cb36
md"In the single-period case, system cost is minimised for any given interval by dispatching the wind generation to its maximum generation limit. As a result, the peaker is used to meet more demanding ramps that cannot be met exclusively by the coal unit. However, the combined ramping capability of the thermal generation is insufficient to mee the ramp from interval 5 to interval 6, thereby resulting in unserved energy"

# ╔═╡ d042585d-4272-4efb-95e0-b9e908ba6382
gen_2 |>
@vlplot(x="interval:n", width=600, height=300) + 
@vlplot(
	mark={:area},
	y={:generation, title="Generation (MW)"},
	color={:generator, legend={title="Generator Type"}},
	title="2-period dispatch"
) +
(
	demandata |>
	@vlplot(
		mark={:line, color=:black,strokeDash=2}, 
		x="interval:n", y=:demand
	)
)

# ╔═╡ b65293ca-3e5a-4c43-981e-21090364e30f
md"With forward-looking ramping constraints, the system uses lower-cost wind to assist in meeting demand ramps. The peaker is operated at its minimum stable level until interval 5, after which it is further ramped to try and help meet demand in interval 6 (N.B. the problems by interval would be: (1,2), (3, 4), (5,6)). However, this is still not enough to ensure that the demand ramp can be entirely satisfied by generation. Note as well that wind is providing both upward and downward flexibility."

# ╔═╡ dbe7597f-7e09-464d-b94f-14bd749d40ea
gen_5 |>
@vlplot(x="interval:n", width=600, height=300) + 
@vlplot(
	mark={:area},
	y={:generation, title="Generation (MW)"},
	color={:generator, legend={title="Generator Type"}},
	title="5-period dispatch"
) +
(
	demandata |>
	@vlplot(
		mark={:line, color=:black,strokeDash=2}, 
		x="interval:n", y=:demand
	)
)

# ╔═╡ dc7e4c61-b75a-4cad-b74c-9b957fce00f4
md"The 5-period dispatch process shows an example of *end-of-horizon* effects/issues. As interval 6 is the start of the next problem (i.e. intervals 1-5 are in one problem, 6-10 in the next), the ramp cannot be catered for entirely through appropriate positioning of generators in intervals 1-5. However, as demand ramps upwards from intervals 3-6, the generators are better placed to handle upward ramping than in the single and two periods problem. This results in less unserved energy"

# ╔═╡ 446fd957-b19d-4015-8190-de6e002d98bc
gen_10 |>
@vlplot(x="interval:n", width=600, height=300) + 
@vlplot(
	mark={:area},
	y={:generation, title="Generation (MW)"},
	color={:generator, legend={title="Generator Type"}},
	title="10-period dispatch"
) +
(
	demandata |>
	@vlplot(
		mark={:line, color=:black,strokeDash=2}, 
		x="interval:n", y=:demand
	)
)

# ╔═╡ 7049d2a2-a331-466d-9a72-a040ee8a0167
md"Finally, if solve all 10 periods simultaneously, generation is appropriately positioned to met the demand and there is no unserved energy. This is achieved by curtailing the wind generator and allowing the coal unit to ramp up in anticipation of the ramp between interval 5 and 6. "

# ╔═╡ b0de9a54-9ef5-433b-8382-b11d0a805add
md"##### Unserved MW"

# ╔═╡ b9eea697-050f-48a3-92f7-3dada6e1ccef
begin
	sumusmws = [sum(use[:, 2]) for use in (usmw_1, usmw_2, usmw_5, usmw_10)]
	usmw = DataFrame(:period=>[1, 2, 5, 10], :usmw=>sumusmws)
	usmw|> @vlplot(
		:bar,
		x={"period:n", title="Dispatch Period Model"}, 
		y={:usmw, title="Unserved MWs (MW/interval)"}, width=600
	)
end

# ╔═╡ 8fe29e7b-c0e9-4ea2-a280-b1a729265ba6
md"As discussed above, the unserved energy (here, we show total unserved MW) decreases as the multi-period horizon is extended."

# ╔═╡ 2a2e78f2-e324-4a5f-be57-ef61fa245d5a
md"##### Pricing"

# ╔═╡ c5836df1-28f0-4baf-a239-8fd5938fc679
md"
Let's recap generator offers:
- Wind - \$$(gens[:Wind].offer)/MW/hr
- Coal - \$$(gens[:Coal].offer)/MW/hr
- Peaker - \$$(gens[:Peaker].offer)/MW/hr
"

# ╔═╡ e6611272-e495-4ea1-bcff-eaf7c83dcb37
md"Below are prices across all of the dispatch processes we've modelled. In each model, I've set the value of lost load at \$15,000/MW/hr. We'll explain below, but you can see that in some cases, multi-period dispatch can result in **negative prices** prior to and high(er) prices following an upward ramping event.

###### A note on pricing
To be precise, prices are the dual of balance constraint in each model. This means they represent the improvement (or additional cost) in the objective function if the constraint were *relaxed* (i.e. RHS value of constraint changed). The relaxation is technically *infinitesimal* - we are talking about a partial derivative. 

However, to simplify how we think about costs and prices below, we use thought experiments where we consider additional benefits and costs if we were to increase demand (i.e. the RHS) by 1 MW. There may be cases where thinking about an additional MW of demand by not be appropriate for determining pricing."

# ╔═╡ b2a1c7ff-d4ea-4847-9bd0-ab8806ebda65
begin
	allprices = copy(prices_1)
	allprices[:, :period] .= 1
	for (i, px) in zip((2, 5, 10), (prices_2, prices_5, prices_10))
		px[:, :period] .= i
		allprices = vcat(allprices, px)
	end
	allprices |>
	@vlplot(
		mark={:line, point=true},
		x={:interval, axis={title="Interval"}}, 
		y={:prices, axis={title="Price (\$/MW/hr)"}}, 
		color={"period:n", axis={title="Dispatch Periods"}},
		title="Prices for all dispatch processes",
		width=450, height=350
		)
end

# ╔═╡ 6c3a4a57-c2a6-423d-8a4c-21f8c7e6aca8
md"###### Single period pricing

As expected, the marginal prices for single period dispatch are set by the marginal generator. This is either coal or the peaker. In interval 6, unserved energy results in price reaching the market cap. "

# ╔═╡ 9b07db72-b51c-49bd-9ac5-da45dc6dadf7
begin
	prices_1 |> 
	@vlplot(width=450, height=350) +
	@vlplot(mark={:line}, x=:interval, y=:prices) +
	@vlplot(
		mark={:text, color=:black, dy=-10, dx=10}, 
		title="Single period prices", text="prices",
		x={"interval", title="Interval"}, y={"prices", title="Price (\$/MW/hr)"},
	)
end

# ╔═╡ 3d2956c3-1365-4fe7-adcf-a180dcb7a2f2
md"###### Two period pricing

Some of the prices reflect the marginal offers (\$0 and \$40), but why \$80 for some intervals? This has to do with multi-dispatch ramping.

Let's have a look at interval 2. It has a price of \$80/MW/hr, which can be thought of as the cost of meeting 1 MW of additional demand in that interval. The additional 1 MW could be met by the coal unit (costing \$40). However, the coal unit is ramp-constrained between intervals 1 and 2 (i.e. operating at its maximum ramp rate). And because intervals 1 and 2 are solved simultaneously and are linked by a ramp constraint, the ramp-constrained coal unit could only feasibly meet 1 MW extra demand in interval 2 if it were generating 1 MW more in interval 1 (at a cost of \$40). 

In short, the optimiser sees the **best solution for meeting additional demand as the coal unit generating 1 MW extra in interval 1 so that it can generate 1 MW extra in interval 2 (due to being ramp-constrained). This results in a total price of \$(40+40) or \$80/MW/hr for interval 2.**
"

# ╔═╡ 48297c90-5595-4b4d-a5e7-002ff52c924d
begin
	prices_2 |> 
	@vlplot(width=450, height=350) +
	@vlplot(mark={:line}, x=:interval, y=:prices) +
	@vlplot(
		mark={:text, color=:black, dy=-10, dx=10}, 
		title="Two period prices", text="prices",
		x={"interval", title="Interval"}, y={"prices", title="Price (\$/MW)"},
	)
end

# ╔═╡ 0b0c52f3-8bc7-4774-aace-60064f602f1e
md" ###### Ten period pricing

We'll skip the five period and move straight on to the ten period process - the only one where prices go negative!

To explain the negative price, let's think again about additional demand in interval 3. Additional demand in interval 3 could be met by increasing generation from the coal unit by 1 MW at a cost of \$40/MW/hr. But that's not where things end - because all the intervals are linked by ramping constraints, we need to consider the knock-on effects of 1 MW more from the coal unit in interval 3. The optimal method of *'using'* this additional generation is to actually have the coal unit dispatched 1 MW higher for intervals 4 and 5 (each at a cost of \$40) so that 1 MW that would otherwise be supplied by the peaker in interval 6 (at a cost of \$1000) can be replaced by a cheaper MW from the coal unit. And because the coal unit is actually ramp-constrained downwards from interval 6 to 7, it must maintain this additional 1 MW generation for that interval as well.

So to summarise:
- If demand were increased by 1 MW in interval 3, an additional 1 MW of coal generation in intervals 3, 4, 5, 6 and 7 would cost $40 × 5$ or an additional \$200.
- However, an additional 1 MW from the coal unit in interval 6 would displace 1 MW from the peaker and therefore cost \$1000 less. We can represent a benefit as a negative cost, i.e. -\$1000.
- Therefore, the net cost is $-1000+200=-800$. So the 'cost' of increasing demand by 1 MW is actually -\$800, which can be intepreted a decrease in cost or benefit. **So, an increase in demand in interval 3 would actually reduce total system costs across the horizon of interest**.

A general pricing outcome of multi-period dispatch when ramp constraints are binding is that the price will often drop prior to an upward ramp and peak at the end of an upward ramp. Despite potential negative pricing, it is still in the generators' best interest to generate as if they withdrew generation, they may not be able to ramp up to capture high prices at the end of the upward ramp. It's also worth noting that the negative pricing is an incentive for demand (response) to increase consumption by 1 MW - you can get paid to use energy in such intervals."

# ╔═╡ e453818b-24bd-4dd6-8391-0c4536c9fbc2
begin
	prices_10 |> 
	@vlplot(width=450, height=350) +
	@vlplot(mark={:line}, x=:interval, y=:prices) +
	@vlplot(
		mark={:text, color=:black, dy=-10, dx=10}, 
		title="Ten period prices", text="prices",
		x={"interval", title="Interval"}, y={"prices", title="Price (\$/MW)"},
	)
end

# ╔═╡ 8dba2e6c-8fd2-4989-9823-0535caebfc41
md"**Confirming the interpretation above**

Below, we actually re-run the model but with an additional 1 MW of demand in interval 3. The chart below confirms that the coal unit generation 1 MW extra in intervals 3-7."

# ╔═╡ 2a39e58a-781f-4ef9-870e-58f765e07117
begin
	demand1 = @SVector[880.0, 1200.0, 1001.0, 1300.0, 1500.0,
	                        2000.0, 1000.0, 800.0, 900.0, 1000.0]
	(prices_11, gen_11, usmw_11) = run_multiperiod(demand1, 10)
	coal_11 = gen_11[gen_11[:, :generator].==Symbol("Coal"), :generation]
	coal_10 = gen_10[gen_10[:, :generator].==Symbol("Coal"), :generation]
	diff_gen = DataFrame(:interval=>1:10, :diff=>(coal_11 .- coal_10))
	diff_gen |> @vlplot(
		:bar, width=450, height=150,
		x={"interval:o", axis={title="Interval"}}, y={:diff, axis={title="MW"}},
		title="Additional generation from coal unit (1 MW additional demand in interval 3)"
	)
end

# ╔═╡ 91b4f5e8-ec60-4ed7-b51b-bde96d964ea8
md"##### Costs

We plot total system costs below, but the comparison is dominated by the costs incurred at market cap (due to unserved energgy in single, two and five period dispatch proceses).
"

# ╔═╡ 2b886b08-4997-4272-982d-f0786adecd4c
begin
	costs = Vector{Float64}(undef, 4)
	for (i, gen, prices) in zip(
		(1, 2, 3, 4), (gen_1, gen_2, gen_5, gen_10), 
		(prices_1, prices_2, prices_5, prices_10)
		)
		costdf = leftjoin(gen, prices, on=:interval)
		cost = sum(costdf[:, :generation] .* (5.0/60.0) .* costdf[:, :prices])
		@inbounds costs[i] = cost
	end
	costs = DataFrame(:periods=>[1, 2, 5, 10], :cost=>costs) 
	costs |> @vlplot(
		:bar, width=350,
		x={"periods:n", title="Dispatch process (no. of intervals)"},
		y={:cost, title="Total cost (\$)"}, title="Total System Cost"
	)
end

# ╔═╡ 9badc690-dce8-44f5-9b01-527c47373081
md"## Some final thoughts

In this notebook, I've tried to highlight some unintuitive pricing outcomes that occur with multi-period dispatch. Though the pricing volatility that may arise from ramping events may impose costs on generators, implementing the solution pricing for all intervals in the horizon will ensure that incentives remain for resources to assist the system in meeting ramps.


Multi-period dispatch can help ensure dispatch feasibility over a horizon, assuming there is sufficient ramping capability in the system and that generation/demand forecasts are accurate. The latter is a big assumption - pricing and dispatch outcomes may no longer be valid if forecasts change. Though a *rolling-horizon approach* can assist in updating dispatch instructions, it's rather unclear how to handle pricing across successive multi-period dispatch processes. This is where procuring reserves comes in - while reserves can impose additional costs to multi-period dispatch (through an additional constraint in the optimisation problem), they can be a simple way to handle uncertainty. For a more in-depth dive and comparison, check out the paper by Ela and Malley linked at the top of this notebook.
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a"

[compat]
DataFrames = "~1.3.2"
GLPK = "~1.0.1"
JuMP = "~0.23.2"
Plots = "~1.27.1"
PlutoUI = "~0.7.37"
StaticArrays = "~1.4.2"
VegaLite = "~2.6.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "c619ac801b5cdbcdbb71eef8ad2bcf41c2c50240"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "c3cc0a7a4e021620f1c0e67679acdbf1be311eb0"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.0.1"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.2.1+2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSONSchema]]
deps = ["HTTP", "JSON", "URIs"]
git-tree-sha1 = "2f49f7f86762a0fbbeef84912265a1ae61c4ef80"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "c48de82c5440b34555cb60f3628ebfb9ab3dc5ef"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "0.23.2"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "a62df301482a41cb7b1db095a4e6949ba7eb3349"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.1.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NodeJS]]
deps = ["Pkg"]
git-tree-sha1 = "905224bbdd4b555c69bb964514cfa387616f0d3a"
uuid = "2bd173c7-0d6d-553b-b6af-13a54713934c"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "1690b713c3b460c955a2957cd7487b1b725878a7"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "995a812c6f7edea7527bb570f0ac39d0fb15663c"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6976fab022fea2ffea3d945159317556e5dad87c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Vega]]
deps = ["DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "JSONSchema", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "Setfield", "TableTraits", "TableTraitsUtils", "URIParser"]
git-tree-sha1 = "43f83d3119a868874d18da6bca0f4b5b6aae53f7"
uuid = "239c3e63-733f-47ad-beb7-a12fde22c578"
version = "2.3.0"

[[deps.VegaLite]]
deps = ["Base64", "DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "TableTraits", "TableTraitsUtils", "URIParser", "Vega"]
git-tree-sha1 = "3e23f28af36da21bfb4acef08b144f92ad205660"
uuid = "112f6efa-9a02-5b7d-90c0-432ed331239a"
version = "2.6.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─bbe08740-1e29-4f63-a755-0bf08d3df898
# ╟─c2e7dc36-b8fc-471c-a6f3-f80aef467eb0
# ╟─249be87e-08a9-49c1-98e8-81a421f1095e
# ╟─85d9814c-8523-4be3-b279-918eef5a55bb
# ╠═05e4f039-9a2a-4197-b25f-0210ce0baa6a
# ╟─0d561b8f-f79e-4587-a602-21c5f6101a8a
# ╟─b4260f67-110c-4d90-92aa-43756fd32ca6
# ╟─4c33ffd2-bda9-4ea9-961b-c9ce9f0821d6
# ╟─99681fe9-8f7e-4681-976b-e44ebb9bcc5b
# ╟─689eae05-e366-4595-8fcd-923e6d95dc16
# ╟─f1591a89-37c6-4f88-99e3-64d8cb415f72
# ╟─537cb82f-825d-46ae-a69a-d8a20493a689
# ╟─6b937c1a-f6e8-4a57-99ab-cd4391f24fef
# ╟─9e76c37f-c4a4-4b12-8f4a-7c12133f19ac
# ╟─3a298fac-8503-46e8-8747-56b93b8564b5
# ╟─9f8913c5-6aa4-4137-9ab7-5f4313371b7d
# ╟─d861def0-9513-4da1-8ee8-a266868c4447
# ╟─ea7ce8b8-f2a6-4fd3-b5aa-71588293b271
# ╟─874cd3d3-2648-4b9f-82cc-c34485b6848f
# ╟─3fca603b-21e6-4aaf-83b6-17b60d8e9218
# ╟─9c29a181-9a29-4fec-85f0-ad87badf0445
# ╟─f00e8a37-a2ba-4744-8abc-675b51f774d4
# ╠═5bd442ad-c171-40f1-ae98-6548667a8b1f
# ╟─56cb2867-4c2c-4f30-b8af-29fae8825f13
# ╟─27ae61bb-5333-4894-9a10-c34ba46ae2f4
# ╟─27c881e5-ad66-42e5-88eb-3b4e28597593
# ╟─dfc1f6f9-b436-4a5c-914b-790dc3e8cb36
# ╟─d042585d-4272-4efb-95e0-b9e908ba6382
# ╟─b65293ca-3e5a-4c43-981e-21090364e30f
# ╟─dbe7597f-7e09-464d-b94f-14bd749d40ea
# ╟─dc7e4c61-b75a-4cad-b74c-9b957fce00f4
# ╟─446fd957-b19d-4015-8190-de6e002d98bc
# ╟─7049d2a2-a331-466d-9a72-a040ee8a0167
# ╟─b0de9a54-9ef5-433b-8382-b11d0a805add
# ╟─b9eea697-050f-48a3-92f7-3dada6e1ccef
# ╟─8fe29e7b-c0e9-4ea2-a280-b1a729265ba6
# ╟─2a2e78f2-e324-4a5f-be57-ef61fa245d5a
# ╟─c5836df1-28f0-4baf-a239-8fd5938fc679
# ╟─e6611272-e495-4ea1-bcff-eaf7c83dcb37
# ╟─b2a1c7ff-d4ea-4847-9bd0-ab8806ebda65
# ╟─6c3a4a57-c2a6-423d-8a4c-21f8c7e6aca8
# ╟─9b07db72-b51c-49bd-9ac5-da45dc6dadf7
# ╟─3d2956c3-1365-4fe7-adcf-a180dcb7a2f2
# ╟─48297c90-5595-4b4d-a5e7-002ff52c924d
# ╟─0b0c52f3-8bc7-4774-aace-60064f602f1e
# ╟─e453818b-24bd-4dd6-8391-0c4536c9fbc2
# ╟─8dba2e6c-8fd2-4989-9823-0535caebfc41
# ╟─2a39e58a-781f-4ef9-870e-58f765e07117
# ╟─91b4f5e8-ec60-4ed7-b51b-bde96d964ea8
# ╟─2b886b08-4997-4272-982d-f0786adecd4c
# ╟─9badc690-dce8-44f5-9b01-527c47373081
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
