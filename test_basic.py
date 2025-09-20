#!/usr/bin/env python3
"""
Test básico para verificar la funcionalidad del módulo sin dependencias pesadas.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_groups_module():
    """Test del módulo de grupos."""
    try:
        from groups import load_groups, get_all_phrases, get_phrases_by_group, get_all_phrases_flat
        print("✅ Módulo groups importado correctamente")

        # Test cargar grupos
        grupos = load_groups()
        print(f"✅ Grupos cargados: {list(grupos.keys())}")

        # Verificar que tenemos 3 grupos
        assert len(grupos) == 3, f"Esperados 3 grupos, encontrados {len(grupos)}"

        # Verificar que cada grupo tiene 10 frases
        for grupo, frases in grupos.items():
            assert len(frases) == 10, f"Grupo {grupo} tiene {len(frases)} frases, esperadas 10"

        print("✅ Test de grupos completado correctamente")

    except Exception as e:
        print(f"❌ Error en test de grupos: {e}")
        return False

    return True


def test_preprocess_module():
    """Test del módulo de preprocesamiento."""
    try:
        from preprocess import normalize_text, preprocess_query, preprocess_phrases
        print("✅ Módulo preprocess importado correctamente")

        # Test normalización de texto
        test_text = "¿Cómo PUEDO crear una CUENTA nueva?!!"
        normalized = normalize_text(test_text)
        expected = "como puedo crear una cuenta nueva"

        assert normalized == expected, f"Esperado: '{expected}', obtenido: '{normalized}'"
        print(f"✅ Normalización correcta: '{test_text}' -> '{normalized}'")

        # Test preprocesamiento de frases
        test_phrases = ["¡Hola Mundo!", "¿Cómo estás?"]
        processed = preprocess_phrases(test_phrases)
        expected_processed = ["hola mundo", "como estas"]

        assert processed == expected_processed, f"Esperado: {expected_processed}, obtenido: {processed}"
        print("✅ Preprocesamiento de frases correcto")

        print("✅ Test de preprocess completado correctamente")

    except Exception as e:
        print(f"❌ Error en test de preprocess: {e}")
        return False

    return True


def test_data_structure():
    """Test de la estructura de datos."""
    try:
        import json

        # Verificar que el archivo JSON existe y es válido
        with open('data/grupos.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'grupos' in data, "Falta clave 'grupos' en el JSON"

        grupos = data['grupos']
        expected_groups = ['A', 'B', 'C']

        for group in expected_groups:
            assert group in grupos, f"Falta grupo {group}"
            assert len(grupos[group]) == 10, f"Grupo {group} no tiene 10 frases"

            # Verificar que todas las frases son strings no vacías
            for i, frase in enumerate(grupos[group]):
                assert isinstance(frase, str), f"Grupo {group}, frase {i} no es string"
                assert frase.strip(), f"Grupo {group}, frase {i} está vacía"

        print("✅ Estructura de datos JSON correcta")
        print(f"   - Grupo A: {len(grupos['A'])} frases")
        print(f"   - Grupo B: {len(grupos['B'])} frases")
        print(f"   - Grupo C: {len(grupos['C'])} frases")

    except Exception as e:
        print(f"❌ Error en test de estructura de datos: {e}")
        return False

    return True


def main():
    """Ejecutar todos los tests."""
    print("🧪 Iniciando tests básicos del módulo PLN...")
    print("=" * 50)

    tests = [
        ("Estructura de datos", test_data_structure),
        ("Módulo de grupos", test_groups_module),
        ("Módulo de preprocesamiento", test_preprocess_module),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n📋 Test: {name}")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ Test '{name}' falló")

    print("\n" + "=" * 50)
    print(f"📊 Resultados: {passed}/{total} tests pasaron")

    if passed == total:
        print("🎉 ¡Todos los tests básicos pasaron correctamente!")
        return True
    else:
        print("⚠️  Algunos tests fallaron")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)