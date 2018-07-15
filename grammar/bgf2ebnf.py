#!/usr/bin/env python
import sys
import re
from bs4 import BeautifulSoup

put = sys.stdout.write

def reformat_nonterminal(name):
    if name.upper() == name:
        return name.lower()
    name = name[0].lower() + name[1:]
    name = re.sub(r'-*([A-Z])', r'_\1', name)
    name = name.replace('-', '_')
    return name.lower()

def parse_recipe(recipe, parent_type):
    if recipe is None:
        raise ValueError("Something went wrong when parsing")
    type_ = recipe.name
    if type_ == 'expression':
        parse_recipe(recipe.findChild(recursive=False), parent_type)
    elif type_ == 'sequence':
        for n, item in enumerate(recipe.findChildren(recursive=False)):
            if n != 0: put(' ')
            parse_recipe(item, type_)
    elif type_ == 'choice':
        if parent_type is None:
            put('\n')
            for n, item in enumerate(recipe.findChildren(recursive=False)):
                put('\t' + ('  ' if n == 0 else '| '))
                parse_recipe(item, type_)
                put('\n')
            put('\t')
        else:
            if parent_type == 'sequence': put('(')
            for n, item in enumerate(recipe.findChildren(recursive=False)):
                if n != 0: put(' | ')
                parse_recipe(item, type_)
            if parent_type == 'sequence': put(')')

    elif type_ == 'plus':
        put('{ ')
        parse_recipe(recipe.findChild(recursive=False), type_)
        put(' }+')
    elif type_ == 'star':
        put('{ ')
        parse_recipe(recipe.findChild(recursive=False), type_)
        put(' }')
    elif type_ == 'optional':
        put('[ ')
        parse_recipe(recipe.findChild(recursive=False), type_)
        put(' ]')
    elif type_ == 'nonterminal':
        name = recipe.contents[0]
        name = reformat_nonterminal(name)
        put(name)
    elif type_ == 'terminal':
        put("'%s'" % recipe.contents[0])
    else:
        raise ValueError("Unrecognized type: %s" % type_)

def parse_grammar(soup):
    for rule in soup.grammar.findChildren('production', recursive=False):
        name = rule.nonterminal.contents[0]
        recipe = rule.expression.findChild()
        put(reformat_nonterminal(name) + ' = ')
        parse_recipe(recipe, None)
        if recipe.name != 'choice': put(' ')
        put(';\n\n')

if __name__ == '__main__':
    soup = BeautifulSoup(open('./fortran90.bgf', 'rb'), 'xml')
    parse_grammar(soup)
