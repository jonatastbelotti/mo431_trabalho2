import numpy as np
import matplotlib.pyplot as plt


class DescidaGradienteRosenbrock():

    MAX_PASSOS = 20000
    TOLERANCIA_MINIMA = 1.0 * (10 ** -4)

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


    def executar(self):
        passo = 1
        ponto = np.array([0, 0, 0])
        ponto_anterior = None
        gradiente = None
        tolerancia = 1000000
        valor = self.func_rosenbrock(ponto[0], ponto[1], ponto[2])
        valor_anterior = None
        self.todos_valores = [valor]


        while passo <= self.MAX_PASSOS and tolerancia >= self.TOLERANCIA_MINIMA:
            # Calculando gradiente
            gradiente = self.calc_gradiente(ponto[0], ponto[1], ponto[2])

            # Calculando novo ponto
            ponto_anterior = ponto
            ponto = ponto - (self.learning_rate * gradiente)

            # Calculando novo valor
            valor_anterior = valor
            valor = self.func_rosenbrock(ponto[0], ponto[1], ponto[2])
            self.todos_valores.append(valor)

            # Calculando tolerância
            tolerancia = self.calc_tolerancia(ponto_anterior, ponto)
            

            print("(%d) x1=%.16f x2=%.16f x3=%.16f tolerancia=%.16f valor=%.16f" % (passo, ponto[0], ponto[1], ponto[2], tolerancia, valor))
            passo += 1


        print("=============================================================")
        print("Passos = %d" % (passo - 1))
        print("x1 = %.16f" % (ponto[0]))
        print("x2 = %.16f" % (ponto[1]))
        print("x3 = %.16f" % (ponto[2]))
        print("valor = %.16f" % (valor))


    def plotar_grafico(self, titulo):
        # Mostrando gráfico
        plt.plot(self.todos_valores)
        plt.title(titulo)
        plt.xlabel("Número de atualizações de $x$")
        plt.ylabel("Valor da função $f(x)$")
        plt.show()

    

    def func_rosenbrock(self, x1, x2, x3):
        resp = 100 * ((x2 - (x1 ** 2)) ** 2)
        resp += (1 - x1) ** 2
        resp += 100 * ((x3 - (x2 ** 2)) ** 2)
        resp += (1 - x2) ** 2

        return resp


    def calc_gradiente(self, x1, x2, x3):
        grad_x1 = None
        grad_x2 = None
        grad_x3 = None

        grad_x1 = 2 * ((200 * (x1 ** 3)) - (200 * x1 * x2) + x1 - 1)

        grad_x2 = (-200 * (x1 ** 2)) + (400 * (x2 ** 3)) + (x2 * (202 - (400 * x3))) -2

        grad_x3 = 200 * (x3 - (x2 ** 2))

        return np.array([grad_x1, grad_x2, grad_x3])

    def calc_tolerancia(self, valor_anterior, valor):
        return np.linalg.norm(valor - valor_anterior) / np.linalg.norm(valor_anterior)
        



# CÓDIGO PRINCIPAL
descida = DescidaGradienteRosenbrock(10 ** -4)
descida.executar()
descida.plotar_grafico("Evolução do valor da função $f(x)$ com $lr = 10^{-4}$")
