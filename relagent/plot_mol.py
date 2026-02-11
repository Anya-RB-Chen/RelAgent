from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image
from IPython.display import display
from rdkit.Chem import rdDepictor
from IPython.display import SVG
import networkx as nx
import matplotlib.pyplot as plt


def show_atom_number(mol, label="atomNote"):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol

def moltosvg(
    mol,
    molSize=(800, 500),
    kekulize=True,
    highlight_atoms=None,
    highlight_atom_colors=None,
    highlight_bonds=None,
    highlight_bond_colors=None,
    bond_width=2.0,
    font_size=16,
    atom_number=True
):
    """Input: mol: RDKit molecule object
              molSize: Size of the image
            kekulize: Kekulize the molecule
            highlight_atoms: List of atom indices to highlight
                Example: [0, 1, 2]
            highlight_atom_colors: List of colors for each atom to highlight (the length should be the same as highlight_atoms)
                Example: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            highlight_bonds: List of bond indices to highlight
                Example: [(0, 1), (1, 2)]
            highlight_bond_colors: List of colors for each bond to highlight (the length should be the same as highlight_bonds)
                Example: [(1, 0, 0), (0, 1, 0)]
            bond_width: Width of the bonds in the drawing
            font_size: Size of the atom label text
    Output: SVG image of the molecule with highlights
    """

    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    if atom_number:
        mc = show_atom_number(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])

    # Set bond width and font size via drawing options
    opts = drawer.drawOptions()
    opts.bondLineWidth = bond_width
    # Fix: RDKit's drawOptions does not allow setting fontSize directly (see error in file_context_0)
    # Instead, set atomLabelFontSize and annotationFontScale if available
    if hasattr(opts, "atomLabelFontSize"):
        opts.atomLabelFontSize = font_size
    if hasattr(opts, "annotationFontScale"):
        # annotationFontScale is a multiplier, so we set it relative to default (1.0)
        opts.annotationFontScale = font_size / 16.0  # 16 is the default in the original code

    bond_indices = []
    bond_colors_dict = {}
    if highlight_bonds:
        for idx, bond in enumerate(highlight_bonds):
            atom1, atom2 = bond
            bond_obj = mc.GetBondBetweenAtoms(atom1, atom2)
            if bond_obj:
                bond_idx = bond_obj.GetIdx()
                bond_indices.append(bond_idx)
                if highlight_bond_colors and idx < len(highlight_bond_colors):
                    bond_colors_dict[bond_idx] = highlight_bond_colors[idx]

    # Process atom highlighting if atoms are provided
    atom_colors_dict = {}
    if highlight_atoms and highlight_atom_colors:
        # highlight_atom_colors should be a dict mapping atom idx to color
        if isinstance(highlight_atom_colors, dict):
            atom_colors_dict = {atom: highlight_atom_colors[atom] for atom in highlight_atoms if atom in highlight_atom_colors}
        else:
            # fallback: assume it's a list, map by index
            atom_colors_dict = {atom: highlight_atom_colors[idx] for idx, atom in enumerate(highlight_atoms) if idx < len(highlight_atom_colors)}

    # Draw the molecule with appropriate highlights
    drawer.DrawMolecule(
        mc,
        highlightAtoms=highlight_atoms if highlight_atoms else None,
        highlightAtomColors=atom_colors_dict if atom_colors_dict else None,
        highlightBonds=bond_indices if highlight_bonds else None,
        highlightBondColors=bond_colors_dict if bond_colors_dict else None
    )

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


relations = ['Containment', 'AtomAttachment', 'BondAttachment', 'Fusion', 'Others']
def plot_graph(graph, relations=relations, title="Molecule Ontology Graph"):
    plt.figure(figsize=(12, 8))
    
    # Use a circular layout for better organization
    pos = nx.circular_layout(graph)
    
    # Draw nodes with improved styling
    nx.draw_networkx_nodes(graph, pos, 
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.7,
                          linewidths=2,
                          edgecolors='navy')
    
    # Draw node labels with better visibility
    nx.draw_networkx_labels(graph, pos, 
                           font_size=10,
                           font_weight='bold',
                           font_color='black')
    
    # Only draw edges for Containment relationship
    edge_lists = []
    edge_labels = {}
    
    for edge in graph.edges(data=True):
        rel = edge[2]['relation']
        if rel in relations:
            edge_lists.append((edge[0], edge[1]))
            edge_labels[(edge[0], edge[1])] = rel
    
    # Draw Containment edges in red
    if edge_lists:
        nx.draw_networkx_edges(graph, pos,
                             edgelist=edge_lists,
                             edge_color='red',
                             width=2,
                             alpha=0.6)
    
    # Draw edge labels with better positioning
    nx.draw_networkx_edge_labels(graph, pos,
                                edge_labels=edge_labels,
                                font_size=8,
                                font_color='darkred',
                                bbox=dict(facecolor='white',
                                         edgecolor='none',
                                         alpha=0.7))
    
    plt.title("Molecule Ontology Graph", fontsize=16, pad=20)
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Create a simple molecule (benzene)
    smiles = "CC1=CC(SC2=C3[Ge](CC(CC)CCCC)(CC(CCCC)CC)C4=C2SC5=C4SC(C6=C(C(N(CCCCCCCC)C7=O)=O)C7=C(C)S6)=C5)=C3S1"
    mol = Chem.MolFromSmiles(smiles)

    highlight_atoms = [2, 1, 52, 51, 3, 4, 5, 6, 
                    24, 25, 26, 27, 28, 29, 30, 50]
    highlight_bonds = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 51), (51, 52), (52, 1),
                        (24, 25), (22, 24), (25, 26), (26, 27), (28, 29), (29, 30), (30, 50), (50, 27)]

    color_hex = "d5d1ff"
    color_rgb = [int(color_hex[i:i+2], 16) for i in (0, 2, 4)]
    color_rgb = (color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255)
    print(color_rgb)
    highlight_atom_colors = [color_rgb for _ in highlight_atoms]
    highlight_bond_colors = [color_rgb for _ in highlight_bonds]

    svg_output = moltosvg(mol, molSize=(800, 500), bond_width=4, font_size=10, atom_number=False,
        highlight_atoms=highlight_atoms, highlight_atom_colors=highlight_atom_colors,
        highlight_bonds=highlight_bonds, highlight_bond_colors=highlight_bond_colors)
    # Save the SVG output to a file so it can be viewed outside of a notebook/terminal
    with open("molecule_test.svg", "w") as f:
        f.write(svg_output)
    print("SVG image saved as molecule_test.svg. Open this file in a browser to view the molecule.")