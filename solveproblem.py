import pandas as pd
import itertools

from pulp import *
from openpyxl import load_workbook


def read_data():
    file_obj = pd.ExcelFile('Ordinamento.xlsx')
    data = pd.read_excel(file_obj, 'data').fillna(0)
    return data


def calculate_switch_costs(cost, data):
    """calculates cost for switching between any two configurations
    """

    index = data.columns
    cost_series = []
    for col in data.columns:
        if col in cost:
            cost_series.append(cost[col])  # TipoColori or Carta
        else:
            cost_series.append(cost[col[0]])  # A or B 
    cost_series = pd.Series(cost_series, index=index)
    
    switch_costs = {}
    for cpc1, row1 in data.iterrows():
        for cpc2, row2 in data.iterrows():
            # cost is zero if rows are identical, else depends on column
            switch_cost = ((~(row1==row2))*1*cost_series).sum()
            switch_costs[(row1.name, row2.name)] = switch_cost    
    return switch_costs


def create_problem_binary(data, switch_costs, source=123, target=127):
    """Create pulp problem to minimize costs, given source and target
    """

    prob = LpProblem('order_problem', LpMinimize)

    graph_edges = list(itertools.permutations(data.index.tolist(), 2))
    nodes = data.index.tolist()

    if source:
        import ipdb; ipdb.set_trace()
        nodes = [source] + [n for n in nodes if n not in [source, target]] + [target]

    var_dict = {}
    for cpc1, cpc2 in graph_edges:
        x=LpVariable(f"choice_{cpc1}_{cpc2}", 0, 1, LpBinary)
        var_dict[(cpc1, cpc2)] = x
    
    dummy_dict = {node: LpVariable(f"dummy_{node}", None, None, LpInteger) for node in nodes}

    prob += lpSum([switch_costs[couple]*var_dict[couple] for couple in var_dict])

    for i, j in graph_edges:
        # each edge only in one direction
        prob += var_dict[i, j] + var_dict[j, i] <= 1

    for node in nodes:
        if node == source:
            # sum of ingoing = 0, sum of outgoing = 1
            prob += lpSum([var_dict[i, j] for i, j in graph_edges if j==node]) == 0
            prob += lpSum([var_dict[j, i] for i, j in graph_edges if j==node]) == 1
        elif node == target:
            # sum of ingoing = 1, sum of outgoing = 0
            prob += lpSum([var_dict[i, j] for i, j in graph_edges if j==node]) == 1
            prob += lpSum([var_dict[j, i] for i, j in graph_edges if j==node]) == 0
        else:
            # sum of ingoing = 1, sum of outgoing = 1
            prob += lpSum([var_dict[i, j] for i, j in graph_edges if j==node]) == 1
            prob += lpSum([var_dict[j, i] for i, j in graph_edges if j==node]) == 1

    # make sure that there are no closed loops 
    # (https://en.wikipedia.org/wiki/Travelling_salesman_problem#Integer_linear_programming_formulation)
    for i in nodes[1:]:
        prob += dummy_dict[i]<=len(nodes)-1
        prob += dummy_dict[i]>=0        
        for j in nodes[1:]:
            if i!=j:
                prob += dummy_dict[i]-dummy_dict[j] + len(nodes)*var_dict[(i, j)] <= len(nodes)-1
    
    return prob, var_dict


def solve_problem(cost, row_from=None, row_until=None, source=None, target=None):
    """solves problem given cost and start-end
    """

    data = read_data()

    if source is None:
        start = data.loc[data['Ordinamento']==row_from].index[0]
        limit = data.loc[data['Ordinamento']==row_until].index[0]
        data = data.loc[start:limit].copy()
    
    data.set_index('CPC', inplace=True)
    data.drop('Ordinamento', axis=1, inplace=True)

    switch_costs = calculate_switch_costs(cost, data)
    prob, var_dict = create_problem_binary(data, switch_costs, source=source, target=target)
    prob.solve()
    print("Status:", LpStatus[prob.status])

    sol = pd.DataFrame(
        [[couple[0], couple[1], var_dict[couple].value()] for couple in var_dict],
        columns=['cpc1', 'cpc2', 'value'])
    sol = sol.loc[sol.value>0]
    
    if source is None:
        source = data.index[0]
    order = [source]

    while len(order)<=len(sol):
        next_value = sol.loc[sol.cpc1==order[-1], 'cpc2'].values[0]
        order.append(next_value)

    print('Ordine ottimale:')
    print(order)
    print('Costo minimo:', prob.objective.value())
    return order


def export_result(order):
    """Exports result to excel
    """

    file_path = 'Ordinamento.xlsx'

    data = read_data()

    order = pd.Series(order).unique()
    to_change = data.loc[data['CPC'].isin(order)]
    remain_same = data.loc[~data['CPC'].isin(order)]
    to_change = to_change.set_index('CPC').loc[order].reset_index().reset_index()
    to_change['index'] = to_change['index']+1

    new_data = remain_same.append(to_change, sort=False)
    new_data = new_data.sort_values('Ordinamento')
    new_data['index'] = new_data['index'].fillna(new_data['Ordinamento']).astype(int)
    
    new_data = new_data.drop('Ordinamento', axis=1)
    new_data = new_data.rename(columns={'index': 'Ordinamento'})

    new_data = new_data[
        ['Ordinamento', 'CPC'] + 
        [col for col in new_data.columns if col not in ['Ordinamento', 'CPC']]]
    new_data = new_data.replace(0, pd.np.nan)

    book = load_workbook(file_path)
    if 'results' in book.get_sheet_names():
        book.remove_sheet(book.get_sheet_by_name('results'))
        book.save(file_path)
        book = load_workbook(file_path)
    
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    writer.book = book

    new_data.to_excel(writer, 'results', index=False)
    writer.save()


if __name__=='__main__':
    
    cost = {'A': 10, 
            'C': 7,
            'Carta': 5, 
            'TipoColori': 7,
            }

    # row_from = 1  # riordina da qui
    # row_until = 15  # riordina fino a 
    
    row_from = int(input('Specificare riga di inizio: '))
    row_until = int(input('Specificare riga di fine: '))

    order = solve_problem(cost, row_from, row_until)
    export_result(order)    

