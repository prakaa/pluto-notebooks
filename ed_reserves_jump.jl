### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 6bf099a5-48cd-466b-8f83-b4c3adedf7be
begin
	# see https://github.com/JuliaPluto/static-export-template
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="DataFrames", version="1"),
        Pkg.PackageSpec(name="GLPK", version="0.14"),
        Pkg.PackageSpec(name="JuMP", version="0.21"),
        Pkg.PackageSpec(name="Plots", version="1"),
    ])
    using DataFrames, GLPK, JuMP, Plots
end

# ╔═╡ 658534ae-b901-11eb-3a20-153b9188b496
md"""
# Economic Dispatch (with Reserves)

In this [Pluto.jl notebook](https://www.youtube.com/watch?v=IAF8DjrQSSk), I have tried to adapt a simple economic dispatch model to demonstrate how reserves (e.g. Frequency Control Ancillary Services or FCAS in Australia's National Electricity Market) might be co-optimised with energy in an energy market. Specifically, this notebook provides an example of how this co-optimisation ienables the *opportunity-cost* of reserve provision to be incorporated into pricing.

The example and code borrows heavily from the economic dispatch tutorial available in [JuMP's documentation](https://jump.dev/JuMP.jl/dev/tutorials/Mixed-integer%20linear%20programs/power_systems/). JuMP is an optimisation framework written for Julia.
"""

# ╔═╡ 23b8e7dc-b611-4ce4-bd3d-5421e24c82b6
md"""
## What is economic dispatch?

If you're not familiar with the term, *economic dispatch* (ED) is basically an optimisation problem that attempts to meet electricity demand with the lowest supply cost. 
- If the system is run by a central utility, this process can help reach the utility decide how to control its generation to meet demand. 
- If the power system has an energy market (from here on, energy is just another word for electricity), generators can *offer* their energy at various prices. In some systems, a market operator runs ED to decide which offers should be accepted to run the system at lowest cost. The process also produces *marginal prices* for energy and reserves, which sets the price that all generators supplying these products receive. This is how we run the energy market in Australia, called the National Electricity Market or NEM.
  - We deal with marginal prices in more detail at the bottom of this notebook, but a simple way of thinking of what they represent is how much will it cost the system to produce an extra MW of energy or reserve? 

ED is an economic/market problem. For example, the constraints we include below do not consider technical requirements and limitations of the power system. However, such limits, or *constraints* can be incorporated and if this is done, we can call the problem *security-constrained economic dispatch*.

## What are reserves?

The examples below run economic dispatch for a single time interval. In reality, ED is run to cover an interval (5 minutes in the NEM). Between one ED process and another: 
- Supply and demand may be mismatched and services may be needed to help correct these (*frequency control*) and/or;
- Certain services may be needed to help with meeting movement in energy supply or demand over a longer term (*ramping* or *operating reserves*)

These could be due to:
- Demand moving up and down in an expected fashion
- Demand forecast errors
- Generators varying their power output in an expected fashion
- Generators tripping unexpectedly
- Renewable energy forecast errors

To provide reserves that can help increase supply (*raise* reserves) or decrease supply (*lower* services), generators may need to be turned up or down in the energy market. In this notebook, we explore how the optimisation problem and its solution accounts for this and ensures that prices incorporate any opportunity-cost.


"""

# ╔═╡ 59ba2527-90a4-45b1-a693-47fad63cddd1
md"""
# Dispatch Modelling
"""

# ╔═╡ 7c118911-43f8-4498-89a7-1ee944b82049
md"""
## Dependencies
"""

# ╔═╡ c5e439bb-434a-4ccb-93c3-45f543dbf86a
md"""
## Hypothetical System

Some major assumptions to simplify this:
- We solve for a single time, so all units of energy expressed in MW
  - No time sequential modelling, which includes ramping constraints
- Assume no transmission losses
- Assume no transmission constraints
- Assume this is a regional, single price market with marginal pricing
- Assume that generator offers reflect short-run marginal costs and that the market accepts only one price-quantity pair (in Australia's National Energy Market, 10 price-quantity pairs are accepted for energy and FCAS)

The 


Furthermore, in terms of generation, we assume:
- Generators 1 and 2 are dispatchable (e.g. coal, CCGT, OCGT)
- There is a third dispatchable generator for the reserve problem
- The wind farms are aggregated into just a single wind unit
"""

# ╔═╡ 72c884ad-18e0-42a6-b615-c5854020ee2f
md"""
![system](https://jump.dev/JuMP.jl/dev/assets/power_systems.png)
"""

# ╔═╡ 6333b70f-6891-45b4-9a2f-7c577ff04cf4
md"""
## Economic Dispatch

Minimise cost of energy supply subject to operational constraints.

``min \sum_{i∈I} c_i^g⋅g_i + c^w⋅w``

where ``I ∈ {1,2}``

subject to:
- ``g_i^{min} ≤ g_i ≤ g_i^{max}`` (min and max power limits)
- ``0 ≤ w ≤ w^f`` (wind power injection ≤ forecast)
- ``∑_{i ∈ I} g_i + w = d^f`` (dispatchable + wind generation = demand forecast)

Where:
- ``g_i`` is the generation output of a dispatchable generator (Generator 1 or 2)

- short-run marginal cost (\$/MW) is given by ``c_i^g``

"""

# ╔═╡ 2a8781db-ec61-4fe7-809f-2d4738fde333
md"""
### JuMP implementation
"""

# ╔═╡ 6e486a98-21b3-4828-a61f-a4ee2e9727a6
function solve_ed(g_max::Array{Float64}, g_min::Array{Float64}, c_g::Array{Float64}, 
				  c_w::Float64, d::Float64, w_f::Float64)
	ed = Model(GLPK.Optimizer)
	@variable(ed, g_min[i] <= g[i = 1:2] <= g_max[i]) # disp. gen. variables
	@variable(ed, 0 <= w <= w_f) # wind variable
	@constraint(ed, [i = 1:2], g[i] ≤ g_max[i]) # max power constraint
	@constraint(ed, [i = 1:2], g[i] ≥ g_min[i]) # min power constraint
	@constraint(ed, w <= w_f) # wind power injection ≤ forecast
	con = @constraint(ed, sum(g) + w == d) # energy balance
	@objective(ed, Min, c_g' * g + c_w * w) # minimise cost
	optimize!(ed)
	return value.(g), value(w), w_f - value(w), objective_value(ed), ed, con
end;

# ╔═╡ 7af097c6-109d-440f-a717-0fae07db5f74
md"""
#### Run ED with input data
"""

# ╔═╡ 85922f92-5d6e-4565-a9c2-2aee77abff9b
begin
	# create input data
	gen_data = DataFrame(Generator=[1, 2], g_max = [1000.0, 1000.0], 
		                 g_min = [300.0, 150.0], c_g = [50.0, 100.0])

	gen_data
end

# ╔═╡ f2572466-4ce7-42f8-a84b-44c529b9a72a
begin
  d = 1500.0
  w_f = 200.0
  c_w = 20.0
end;

# ╔═╡ 164852aa-b904-4039-b851-abc40dccd046
md"""

Cost of wind: \$ $(c_w) / MW

Wind forecast: $w_f MW

Demand: $d MW
"""

# ╔═╡ eb1d1635-b8c4-4816-b022-8a695922f51d
md"""
#### Run model
"""

# ╔═╡ 79b8d28c-3ad7-4726-9d65-955ceaf0f2b7
(g_opt, w_opt, ws_opt, obj, model, con) = solve_ed(gen_data.g_max, gen_data.g_min,
								         	   	   gen_data.c_g, c_w, d, w_f);

# ╔═╡ b026eda0-c5b1-4c63-a755-97fdd06a8203
md"""
Check if optimal and check for duals. The latter is important for marginal prices.
"""

# ╔═╡ f1b110d5-fed9-4112-8c72-7469f6a80d36
termination_status(model)

# ╔═╡ 31d6d379-497f-4f84-b4bd-46678a0be787
has_duals(model)

# ╔═╡ a22df840-aa3a-4bac-99cc-50bdbacadee8
md"""
Primal feasible and has duals. The latter is important for calculating market prices.
"""

# ╔═╡ 89f9610b-7495-4790-b0f3-0327a5b6dd65
md"""

#### Results
For a demand of $d MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_opt[1]) MW
  - ``g_2``: $(g_opt[2]) MW
  - Wind: $w_opt MW, spilled = $ws_opt MW

Total cost: \$ $obj

Shadow price of energy balance constraint (supply = demand): $(shadow_price(con)), so marginal cost is \$$(shadow_price(con)*-1)/MW. This is the price of energy.


See end for an explanation of the marginal price.

-----
"""

# ╔═╡ e66ffc80-74e5-43ae-a56d-c94494087410
md"""
#### Reduced demand ED
Now run a model with **demand at 600 MW**.
"""

# ╔═╡ 205fffe4-9cd6-4092-929f-a3ccf0434ff6
md"""
##### A note about modifying JuMP models
If a constraint or the objective function is changed, it is faster to modify specific constraint or objective function to reduce computational burden or rebuilding the model. 

For example, modifying the demand can be done by redefining the constraint and resolving, rather than specifying an entirely new model."""

# ╔═╡ 31843b48-0c7c-487f-ac38-d083e7de7a6a
begin
	d_new = 600.0
	set_normalized_rhs(con, d_new)
	optimize!(model)
	g_new = value.(model[:g])
	w_new = value(model[:w])
	obj_new = objective_value(model)
end;

# ╔═╡ c8de4535-f725-485c-961c-cea6744d4cd0
md"""
Check if optimal and check for duals. The latter is important for marginal prices.
"""

# ╔═╡ 32e90f3b-3735-45a3-bea1-336f2751ec63
termination_status(model)

# ╔═╡ ee70e6cb-4bc0-45f7-8168-3a76d12c2653
has_duals(model)

# ╔═╡ 3eebdc9b-8132-43b2-96fa-362bee722dee
md"""
#### Results
For a demand of $d_new MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_new[1]) MW
  - ``g_2``: $(g_new[2]) MW
  - Wind: $w_new MW, spilled = $(w_f-w_new) MW

Total cost: \$ $obj_new

Shadow price of energy balance constraint (supply = demand): $(shadow_price(con)), so marginal cost is \$$(shadow_price(con)*-1)/MW. This is the price of energy.

In this case, wind energy is spilled because minimum generation constraints must be met for Generators 1 and 2.

-----

"""

# ╔═╡ 70c617cf-c854-4f94-8851-11d7fce94dce
md"""
#### ED across various demand scenarios
We will next run the model across various demand scenarios, from demand = 100 MW to demand = 3000 MW. Only feasible solutions are plotted in the chart below.
"""

# ╔═╡ 703dced5-ed66-4773-8f1b-0d6ea31e5cb7
function modify_demand_and_optimise(d_con::ConstraintRef, d_new::Float64, 											    model::Model)
	set_normalized_rhs(d_con, d_new)
	optimize!(model)
	g_new = value.(model[:g])
	w_new = value(model[:w])
	obj_new = objective_value(model)
	if termination_status(model) == MOI.OPTIMAL
		price = shadow_price(d_con) * -1
		return [obj_new, g_new, w_new, d_new, price]
	else
		return repeat([nothing], 5)
	end
end;

# ╔═╡ 20b650f6-8419-45a9-87bf-febca5f3e03d
begin
	possible_demands = 100.0:100.0:3000.0
	obj_vec = Float64[]
	g_vec = Array{Float64}[]
	w_vec = Float64[]
	d_vec = Float64[]
	price_vec = Float64[]
	md"""
	Initialise iteration
	"""
end

# ╔═╡ 6d81b532-ce59-4d95-8641-04e0b474a330
for demand in possible_demands
	results = modify_demand_and_optimise(con, demand, model)
	if all(isnothing.(results))
		continue
	else
		push!(obj_vec, results[1])
		push!(g_vec, results[2])
		push!(w_vec, results[3])
		push!(d_vec, results[4])
		push!(price_vec, results[5])
	end
end

# ╔═╡ 1aca7641-2786-4bcb-b28f-590eb1b3434c
begin
	g1 = [x[1] for x in g_vec]
	g2 = [x[2] for x in g_vec]
	l = @layout [a{1w}; b{1w}]
	p1 = bar(d_vec, g1, label="Generator 1", legend=:outerright)
	bar!(d_vec, g2, label="Generator 2", legend=:outerright)
	bar!(d_vec, w_vec, label="Wind",  legend=:outerright)
	ylabel!("Generation (MW)")
	p2 = bar(d_vec, price_vec, label="Energy Price", color=:brown, 
			 legend=:outerright)
	ylabel!("Energy Price (\$/MW)")
	xlabel!("Energy Demand (MW)")
	plot(p1, p2, layout=l)
end

# ╔═╡ 6a158311-f151-40aa-b18c-9254378a6429
md"""
## Economic Dispatch with Reserves

Now we will include reserves into the optimisation problem (co-optimisation). We will assume that only dispatchable generators (Generators 1, 2 and 3) can provide reserves.

Minimise cost of energy supply and reserves subject to operational constraints.

``min \sum_{i∈I} c_i^g⋅g_i + \sum_{i∈I} c_i^r⋅r_i + c^w⋅w``

where ``I ∈ {1,2,3}``

subject to:
- ``g_i^{min} ≤ g_i`` (reserve and generation must be within capacity limits)
- ``g_i + r_i ≤ g_i^{max}`` (reserve and generation must be within capacity limits)
- ``0 ≤ w ≤ w^f`` (wind power injection ≤ forecast)
- ``∑_{i ∈ I} r_i = R`` (reserve requirement met)
- ``∑_{i ∈ I} g_i + w = d^f`` (dispatchable + wind generation = demand forecast)

Where:
- ``g_i`` is the generation output of a dispatchable generator (Generator 1, 2 or 3)

- short-run marginal cost of energy (\$/MW) is given by ``c_i^g``
- ``r_i`` is the reserve provided by a dispatchable generator (Generator 1, 2 or 3)

- short-run marginal cost of reserve (\$/MW) is given by ``r_i^g``

"""

# ╔═╡ ed843abd-9a1d-442f-913a-473dc6d47bef
md"""
### JuMP Implementation
"""

# ╔═╡ a8a63601-eeef-4c7d-8b43-c651cf4bbbf8
function solve_ed_with_reserves(g_max::Array{Float64}, g_min::Array{Float64}, 
								c_g::Array{Float64}, c_w::Float64, 
								c_r::Array{Float64}, R::Float64, d::Float64,
								w_f::Float64)
	ed = Model(GLPK.Optimizer)
	@variable(ed, g_min[i] <= g[i = 1:3] <= g_max[i]) # disp. gen. variables
	@variable(ed, g_min[i] <= r[i = 1:3] <= g_max[i]) # disp. reserve variables
	@variable(ed, 0 <= w <= w_f) # wind power injection variable
	@constraint(ed, [i = 1:3], g[i] + r[i] ≤ g_max[i]) # max power constraint
	@constraint(ed, [i = 1:3], g[i] ≥ g_min[i]) # min power constraint
	@constraint(ed, w <= w_f) # wind injection ≤ forecast
	en_con = @constraint(ed, sum(g) + w == d) # energy balance
	r_con = @constraint(ed, sum(r) == R) # reserve requirement
	@objective(ed, Min, c_g' * g + c_w * w + c_r' * r)
	optimize!(ed)
	return value.(g), value.(r), value(w), objective_value(ed), en_con, r_con, model
end

# ╔═╡ 9a7c3e64-641e-48b8-9c5f-2eb0a1e5727f
begin
	reserve_gen_data = copy(gen_data)
	push!(reserve_gen_data, [3, 500.0, 0.0, 300.0])
	reserve_gen_data[!, :c_r] = [40.0, 80.0, 500.0]
	reserve_gen_data
end

# ╔═╡ b73b8cc0-bcb6-4a96-ba68-6bd898305760
md"""
Cost of wind: \$ $(c_w) / MW

Wind forecast: $w_f MW

Demand: $d MW

**Reserve requirement = 701 MW**
- Arbitrary for an interesting example!

"""

# ╔═╡ 882ddb8c-a8eb-42cc-8f53-03723875608a
R = 701.0

# ╔═╡ 594d0f3f-ae19-4736-a73b-9067b182812c
md"""
#### Run model
"""

# ╔═╡ be45d9fc-c539-4cde-babd-92c03bc7f044
(g_r, r_r, w_r, obj_r, en_con, r_con, r_mod) =
	solve_ed_with_reserves(reserve_gen_data.g_max,
						   reserve_gen_data.g_min,
						   reserve_gen_data.c_g,
						   c_w, reserve_gen_data.c_r,
						   R, d, w_f);

# ╔═╡ 62d03d73-caa3-430e-8d97-014166a0ac5c
md"""
Check if optimal and check for duals. The latter is important for marginal prices.
"""

# ╔═╡ a6695f34-24b0-4cf9-915a-1be3ae8ac105
termination_status(r_mod)

# ╔═╡ 3872a97b-5760-4ee3-88b5-adc22a42c34b
has_duals(r_mod)

# ╔═╡ af9e70d2-2fba-436c-8f82-9e8f941074a6
md"""
Primal feasible and has duals. We need the duals to find the marginal cost/shadow price.
"""

# ╔═╡ 6d70c9f6-d708-4d42-ab23-e449edc6f895
md"""
#### Results
For a demand of $d MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_r[1]) MW, ``r_1``: $(r_r[1]) MW
  - ``g_2``: $(g_r[2]) MW, ``r_2``: $(r_r[2]) MW
  - ``g_3``: $(g_r[3]) MW, ``r_3``: $(r_r[3]) MW
  - Wind: $w_r MW, spilled = $(w_f-w_r) MW

Total cost: \$ $obj_r

Shadow price of energy balance constraint (supply = demand): $(shadow_price(en_con)), so marginal cost is \$$(shadow_price(en_con)*-1)/MW. This is the price of energy.

Shadow price of reserve requirement constraint (∑r = R): $(shadow_price(r_con)), so marginal cost is \$$(shadow_price(r_con)*-1)/MW. This is the price of reserves.

"""

# ╔═╡ 9778a606-ff49-4ce9-927c-c5cd259503e0
md"""
#### Reserve price explanation

Why is the reserve price 280 when reserve offers are 40, 80 or 500?

##### Shadow prices and marginal prices

Marginal prices for energy and reserves are essentially the price to service an infinitesimal increase in demand. For our thought experiment, we will simplify this by thinking of the cost to service the next MW of demand or reserve.

###### Formal and simpler definition
The marginal price for energy is formally the *shadow price* of the supply-demand balance constraint (also known as the value of the Lagrange multiplier, or the value of the dual variable of the constraint at the optimal value of the dual problem). Similarly, the marginal price for reserves is the shadow price of the reserve requirement constraint. In this case, the shadow price represents the additional cost of the objective function if the constraint is relaxed. 

Simply put, a shadow price is effectively the *total additional cost to the system* to supply the next MW of energy or reserve.

##### Marginal price of energy
In the situation above, it is cheapest to service the next MW of energy by turning up Generator 3, so the marginal cost of energy is 300. 

##### Marginal price of reserve
However, for the next MW of reserve, it is actually cheapest to turn Generator 2's energy production down 1 MW and thereby obtain 1 MW of reserve from Generator 2. However, to ensure that energy supply and demand and balanced, Generator 3 must be turned up 1 MW. Obtaining reserve from Generator 2 costs 80 (\$c^r_2\$), turning Generator 2 down 1 MW in energy "costs" -100 (turning it down actually reduces the total cost) and turning Generator 3 up 1 MW in energy costs 300, giving us a total of 280. Hence the *total* cost to the system to increase reserves by 1 MW has been accounted for.

###### Participant's perspective
From the perspective of Generator 2, co-optimisation ensures that opporunity-cost in the energy market is accounted for. Since the price for reserves is set by Generator 2, the marginal price is the sum of its reserve offer cost and its opporunity-cost in the energy market:
1. The reserve offer is 80.
2. By being turned down 1 MW in the energy market, Generator 2 misses out on a profit (opportunity cost). This is:
    - (Price of energy) - (Generator 2's short run marginal cost, or assumed to be anyway), which in this case is (300)-(100) = 200. 
The sum of these is 280, the price of reserves. Simply put, if a unit is backed off energy to provide reserves (e.g. FCAS), the optimisation will ensure that it does not miss out on any profits so long as its bids reflect short run marginal costs.
"""

# ╔═╡ Cell order:
# ╟─658534ae-b901-11eb-3a20-153b9188b496
# ╟─23b8e7dc-b611-4ce4-bd3d-5421e24c82b6
# ╟─59ba2527-90a4-45b1-a693-47fad63cddd1
# ╟─7c118911-43f8-4498-89a7-1ee944b82049
# ╠═6bf099a5-48cd-466b-8f83-b4c3adedf7be
# ╟─c5e439bb-434a-4ccb-93c3-45f543dbf86a
# ╟─72c884ad-18e0-42a6-b615-c5854020ee2f
# ╟─6333b70f-6891-45b4-9a2f-7c577ff04cf4
# ╟─2a8781db-ec61-4fe7-809f-2d4738fde333
# ╠═6e486a98-21b3-4828-a61f-a4ee2e9727a6
# ╟─7af097c6-109d-440f-a717-0fae07db5f74
# ╟─85922f92-5d6e-4565-a9c2-2aee77abff9b
# ╠═f2572466-4ce7-42f8-a84b-44c529b9a72a
# ╟─164852aa-b904-4039-b851-abc40dccd046
# ╟─eb1d1635-b8c4-4816-b022-8a695922f51d
# ╠═79b8d28c-3ad7-4726-9d65-955ceaf0f2b7
# ╟─b026eda0-c5b1-4c63-a755-97fdd06a8203
# ╠═f1b110d5-fed9-4112-8c72-7469f6a80d36
# ╠═31d6d379-497f-4f84-b4bd-46678a0be787
# ╟─a22df840-aa3a-4bac-99cc-50bdbacadee8
# ╟─89f9610b-7495-4790-b0f3-0327a5b6dd65
# ╟─e66ffc80-74e5-43ae-a56d-c94494087410
# ╟─205fffe4-9cd6-4092-929f-a3ccf0434ff6
# ╠═31843b48-0c7c-487f-ac38-d083e7de7a6a
# ╟─c8de4535-f725-485c-961c-cea6744d4cd0
# ╠═32e90f3b-3735-45a3-bea1-336f2751ec63
# ╠═ee70e6cb-4bc0-45f7-8168-3a76d12c2653
# ╟─3eebdc9b-8132-43b2-96fa-362bee722dee
# ╟─70c617cf-c854-4f94-8851-11d7fce94dce
# ╠═703dced5-ed66-4773-8f1b-0d6ea31e5cb7
# ╟─20b650f6-8419-45a9-87bf-febca5f3e03d
# ╠═6d81b532-ce59-4d95-8641-04e0b474a330
# ╟─1aca7641-2786-4bcb-b28f-590eb1b3434c
# ╟─6a158311-f151-40aa-b18c-9254378a6429
# ╟─ed843abd-9a1d-442f-913a-473dc6d47bef
# ╠═a8a63601-eeef-4c7d-8b43-c651cf4bbbf8
# ╟─9a7c3e64-641e-48b8-9c5f-2eb0a1e5727f
# ╟─b73b8cc0-bcb6-4a96-ba68-6bd898305760
# ╠═882ddb8c-a8eb-42cc-8f53-03723875608a
# ╟─594d0f3f-ae19-4736-a73b-9067b182812c
# ╠═be45d9fc-c539-4cde-babd-92c03bc7f044
# ╟─62d03d73-caa3-430e-8d97-014166a0ac5c
# ╠═a6695f34-24b0-4cf9-915a-1be3ae8ac105
# ╠═3872a97b-5760-4ee3-88b5-adc22a42c34b
# ╟─af9e70d2-2fba-436c-8f82-9e8f941074a6
# ╟─6d70c9f6-d708-4d42-ab23-e449edc6f895
# ╟─9778a606-ff49-4ce9-927c-c5cd259503e0
